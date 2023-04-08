import torch.nn as nn
from torch import Tensor
from typing import Tuple
from builder.models.src.transformer.attention import MultiHeadAttention
from builder.models.src.transformer.module import PositionalEncoding, FeedForward, LayerNorm, FeedForwardUseConv
from builder.models.src.transformer.utils import *
from builder.models.src.transformer.encoder import TransformerEncoderLayer


class TrimodalTransformerEncoder_Multitokens_MBT(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
            batch_size: int,
            n_modality: int,
            bottlenecks_n: int,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            txt_idx: int = 2,
            resbottle: bool = False,
            use_pe: list = [False, False, True],
            mask: list = [True, True, True]):
        super(TrimodalTransformerEncoder_Multitokens_MBT, self).__init__()

        self.use_pe = use_pe
        self.n_modality = n_modality
        self.fusion_idx = fusion_startidx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.bottlenecks_n = bottlenecks_n
        self.mask = mask
        self.resbottle = resbottle
        
        self.idx_order = torch.arange(0, batch_size).type(torch.LongTensor)
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        # CLASSIFICATION TOKENS
        self.final_cls_tokens_num = [4, 2, 2]
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, self.final_cls_tokens_num[idx], d_model)) for idx in range(n_modality)])
        
        # mbs (multi bottlenecks)
        self.bottlenecks_vit = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))
        self.bottlenecks_vi = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))
        self.bottlenecks_vt = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))
        self.bottlenecks_it = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))
        self.bottlenecks = nn.ParameterList([self.bottlenecks_vit, self.bottlenecks_vi, self.bottlenecks_vt, self.bottlenecks_it])
        self.bottlenecks_map = [[0,1,2], [0,1,3], [0,2,3]]
        
        # self.v_bottleneck_mask = torch.zeros(36 + self.final_cls_tokens_num[0], 36 + self.final_cls_tokens_num[0]).cuda()              
        self.v_bottleneck_mask = torch.zeros(16,16).cuda()              
        self.v_bottleneck_mask[:16, :16] = 1
        self.v_bottleneck_mask[:4, :4] = 0
        self.v_bottleneck_mask[4:8, 4:8] = 0
        self.v_bottleneck_mask[8:12, 8:12] = 0
        self.v_bottleneck_mask[12, 12] = 0
        self.v_bottleneck_mask[13, 13] = 0
        self.v_bottleneck_mask[14, 14] = 0
        self.v_bottleneck_mask[15, 15] = 0
        self.v_bottleneck_mask[12,:4] = 0
        self.v_bottleneck_mask[:4,12] = 0
        self.v_bottleneck_mask[13,4:8] = 0
        self.v_bottleneck_mask[4:8, 13] = 0
        self.v_bottleneck_mask[14, 8:12] = 0
        self.v_bottleneck_mask[8:12, 14] = 0
        self.v_bottleneck_mask = self.v_bottleneck_mask.ge(0.5)
        
        self.i_bottleneck_mask = torch.zeros(61 + self.final_cls_tokens_num[1], 61 + self.final_cls_tokens_num[1]).cuda()              
        self.i_bottleneck_mask[:16, :16] = 1
        self.i_bottleneck_mask[:4, :4] = 0
        self.i_bottleneck_mask[4:8, 4:8] = 0
        self.i_bottleneck_mask[8:12, 8:12] = 0
        self.i_bottleneck_mask[12, 12] = 0
        self.i_bottleneck_mask[13, 13] = 0
        self.i_bottleneck_mask[12, :4] = 0
        self.i_bottleneck_mask[:4, 12] = 0
        self.i_bottleneck_mask[13, 4:8] = 0
        self.i_bottleneck_mask[4:8,13] = 0
        self.i_bottleneck_mask = self.i_bottleneck_mask.ge(0.5)
        
        self.t_bottleneck_mask = torch.zeros(140 + self.final_cls_tokens_num[2], 140 + self.final_cls_tokens_num[2]).cuda()              
        self.t_bottleneck_mask[:16, :16] = 1
        self.t_bottleneck_mask[:4, :4] = 0
        self.t_bottleneck_mask[4:8, 4:8] = 0
        self.t_bottleneck_mask[8:12, 8:12] = 0
        self.t_bottleneck_mask[12, 12] = 0
        self.t_bottleneck_mask[13, 13] = 0
        self.t_bottleneck_mask[12, :4] = 0
        self.t_bottleneck_mask[:4, 12] = 0
        self.t_bottleneck_mask[13, 4:8] = 0
        self.t_bottleneck_mask[4:8,13] = 0
        self.t_bottleneck_mask = self.t_bottleneck_mask.ge(0.5)
        self.bottleneck_mask_list = [self.v_bottleneck_mask, self.i_bottleneck_mask, self.t_bottleneck_mask]
        
        # key = missing case           (the key = 3, 4th item is ignored)
        # item = bottleneck mean case
        self.b_out_mean_map = {0:[torch.Tensor([0,1,2]).to(torch.long), torch.Tensor([0,1]).to(torch.long), torch.Tensor([0,2]).to(torch.long), torch.Tensor([0]).to(torch.long)], 
                                        1:[torch.Tensor([0,1]).to(torch.long), torch.Tensor([0,1]).to(torch.long), torch.Tensor([0]).to(torch.long), torch.Tensor([0]).to(torch.long)], 
                                        2:[torch.Tensor([0,1]).to(torch.long), torch.Tensor([0]).to(torch.long), torch.Tensor([0,1]).to(torch.long), torch.Tensor([0]).to(torch.long)], 
                                        3:[torch.Tensor([0,1]).to(torch.long), torch.Tensor([0]).to(torch.long), torch.Tensor([1]).to(torch.long), torch.Tensor([0]).to(torch.long)]}
                                                                                
        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None, missing = None):
        cls_token_per_modality = [cls_token.repeat(enc_outputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks_list = [bottleneck_token.repeat(enc_outputs[0].size(0), 1, 1) for bottleneck_token in self.bottlenecks]
        bottleneck_mask_list = [bottleneck_mask.repeat(enc_outputs[0].size(0), 1, 1) for bottleneck_mask in self.bottleneck_mask_list]
        enc_inputs = [torch.cat([cls_token_per_modality[idx], enc_input], axis=1) for idx, enc_input in enumerate(enc_outputs)]
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        for n_modal in range(self.n_modality):
            varying_lengths[n_modal] += self.final_cls_tokens_num[n_modal]
            fixed_lengths[n_modal] += self.final_cls_tokens_num[n_modal]
            if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
            if self.mask[n_modal]:
                self_attn_masks.append(get_attn_pad_mask(enc_inputs[n_modal], varying_lengths[n_modal], fixed_lengths[n_modal]))
            else:
                self_attn_masks.append(None)
                
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx
            
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx]) +
                    self.positional_encoding(enc_inputs[idx].size(1))
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            if idx < self.fusion_idx:
                for modal_idx, enc_layer in enumerate(enc_layers):
                    enc_output, _ = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx])
                    enc_outputs.append(enc_output)      
                    
            else:
                bottleneck_outputs = [[], [], [], []]
                if self.resbottle:
                    res_bottles = list(bottlenecks_list)

                for modal_idx, enc_layer in enumerate(enc_layers):
                    bottlenecks = torch.cat([bottlenecks_list[i] for i in self.bottlenecks_map[modal_idx]], axis=1)
                    b_enc_output = torch.cat([bottlenecks, enc_inputs[modal_idx]], axis=1)
                    if len(bottleneck_self_attn_masks) < self.n_modality:
                        if self.mask[modal_idx]:
                            b_mask = get_attn_pad_mask(b_enc_output, varying_lengths[modal_idx]+bottlenecks.size(1), b_enc_output.size(1)).clone().detach()
                            if modal_idx == 0:
                                b_mask[:,:16,:16] = bottleneck_mask_list[modal_idx][:,:,:]
                            else:
                                b_mask = b_mask + bottleneck_mask_list[modal_idx]
                            bottleneck_self_attn_masks.append(b_mask)
                        else:
                            bottleneck_self_attn_masks.append(None)
                            
                    enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                    
                    for i in range(3):
                        bottleneck_outputs[self.bottlenecks_map[modal_idx][i]].append(enc_output[:, self.bottlenecks_n*i : self.bottlenecks_n*(i+1), :])
                    enc_output = enc_output[:, self.bottlenecks_n*3:, :]
                    enc_outputs.append(enc_output)
                
                bottlenecks_list = []
                for b_idx, b_output in enumerate(bottleneck_outputs):                
                    b_output_stack = torch.stack(b_output)
                    all_possible_b_means = torch.stack([torch.mean(b_output_stack[i, :, :, :], dim=0) for i in self.b_out_mean_map[b_idx]])
                    bottle_output = all_possible_b_means[missing, self.idx_order, :, :]
                    if self.resbottle:
                        bottle_output = torch.mean(torch.stack([bottle_output,res_bottles[b_idx]]), dim=0)
                        # bottle_output = bottle_output + res_bottles[b_idx]
                    bottlenecks_list.append(bottle_output)

        return enc_outputs, 0

class BimodalTransformerEncoder_MBT(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
            batch_size: int,
            n_modality: int,
            bottlenecks_n: int,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 10000,
            txt_idx: int = 2,
            mbt_bottlenecks_type: str = 'skip',
            use_pe: list = [True, True],
            mask: list = [True, True]):
        super(BimodalTransformerEncoder_MBT, self).__init__()

        self.mbt_bottlenecks_type = mbt_bottlenecks_type
        self.use_pe = use_pe
        self.n_modality = 2
        self.fusion_idx = fusion_startidx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.bottlenecks_n = bottlenecks_n
        self.mask = mask
        
        self.idx_order = torch.range(0, batch_size-1).type(torch.LongTensor)
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
                
        # CLASSIFICATION TOKENS
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        self.bottlenecks = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))

        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None, missing = None):
        cls_token_per_modality = [cls_token.repeat(enc_outputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks = self.bottlenecks.repeat(enc_outputs[0].size(0), 1, 1)
        enc_inputs = [torch.cat([cls_token_per_modality[idx], enc_input], axis=1) for idx, enc_input in enumerate(enc_outputs)]
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        for n_modal in range(self.n_modality):
            varying_lengths[n_modal] += 1
            fixed_lengths[n_modal] += 1
            if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
            if self.mask[n_modal]:
                self_attn_masks.append(get_attn_pad_mask(enc_inputs[n_modal], varying_lengths[n_modal], fixed_lengths[n_modal]))
            else:
                self_attn_masks.append(None)
                
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx
            
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx]) +
                    self.positional_encoding(enc_inputs[idx].size(1))
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            # if idx < self.fusion_idx:
            #     for modal_idx, enc_layer in enumerate(enc_layers):
            #         enc_output, _ = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx])
            #         enc_outputs.append(enc_output)      
                    
            # else:
            bottleneck_outputs = []
            for modal_idx, enc_layer in enumerate(enc_layers):
                b_enc_output = torch.cat([bottlenecks, enc_inputs[modal_idx]], axis=1) #bottleneck, cls, input
                if len(bottleneck_self_attn_masks) < self.n_modality:
                    if self.mask[modal_idx]:
                        b_mask = get_attn_pad_mask(b_enc_output, varying_lengths[modal_idx]+self.bottlenecks_n, b_enc_output.size(1))
                        bottleneck_self_attn_masks.append(b_mask)
                    else:
                        bottleneck_self_attn_masks.append(None)
                enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                bottleneck_outputs.append(enc_output[:, :self.bottlenecks_n, :])
                enc_output = enc_output[:, self.bottlenecks_n:, :]
                enc_outputs.append(enc_output)
                
            bottleneck_outputs_stack = torch.stack(bottleneck_outputs)
            bottlenecks_bi_mean = torch.mean(bottleneck_outputs_stack, dim=0)
            all_bottleneck_stack = torch.stack([bottlenecks_bi_mean, bottleneck_outputs_stack[0,:,:,:]])

            bottlenecks = all_bottleneck_stack[missing, self.idx_order, :, :]
            
        return enc_outputs, 0

class TrimodalTransformerEncoder_MBT(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
            batch_size: int,
            n_modality: int,
            bottlenecks_n: int,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 10000,
            resbottle: bool = False,
            txt_idx: int = 2,
            mbt_bottlenecks_type: str = 'skip',
            use_pe: list = [True, True, True],
            mask: list = [True, False, True]):
        super(TrimodalTransformerEncoder_MBT, self).__init__()

        self.mbt_bottlenecks_type = mbt_bottlenecks_type
        self.use_pe = use_pe
        self.n_modality = n_modality
        self.fusion_idx = fusion_startidx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.bottlenecks_n = bottlenecks_n
        self.mask = mask
        self.resbottle = resbottle
        
        self.idx_order = torch.arange(0, batch_size).type(torch.LongTensor)
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
                
        # CLASSIFICATION TOKENS
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        self.bottlenecks = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))

        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None, missing = None):
        cls_token_per_modality = [cls_token.repeat(enc_outputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks = self.bottlenecks.repeat(enc_outputs[0].size(0), 1, 1)
        enc_inputs = [torch.cat([cls_token_per_modality[idx], enc_input], axis=1) for idx, enc_input in enumerate(enc_outputs)]
        
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        for n_modal in range(self.n_modality):
            varying_lengths[n_modal] += 1
            fixed_lengths[n_modal] += 1
            if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
            if self.mask[n_modal]:
                # print("1 ", enc_inputs[n_modal].shape)
                # print("2 ", varying_lengths[n_modal])
                # print("3 ", fixed_lengths[n_modal])
                self_attn_masks.append(get_attn_pad_mask(enc_inputs[n_modal], varying_lengths[n_modal], fixed_lengths[n_modal]))
            else:
                self_attn_masks.append(None)
                
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx
            
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx]) +
                    self.positional_encoding(enc_inputs[idx].size(1))
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            if idx < self.fusion_idx:
                for modal_idx, enc_layer in enumerate(enc_layers):
                    enc_output, _ = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx])
                    enc_outputs.append(enc_output)      
                    
            else:
                bottleneck_outputs = []
                if self.resbottle:
                    res_bottles = bottlenecks

                for modal_idx, enc_layer in enumerate(enc_layers):
                    b_enc_output = torch.cat([bottlenecks, enc_inputs[modal_idx]], axis=1) #bottleneck, cls, input
                    if len(bottleneck_self_attn_masks) < self.n_modality:
                        if self.mask[modal_idx]:
                            b_mask = get_attn_pad_mask(b_enc_output, varying_lengths[modal_idx]+self.bottlenecks_n, b_enc_output.size(1))
                            bottleneck_self_attn_masks.append(b_mask)
                        else:
                            bottleneck_self_attn_masks.append(None)
                            
                    enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                    bottleneck_outputs.append(enc_output[:, :self.bottlenecks_n, :])
                    enc_output = enc_output[:, self.bottlenecks_n:, :]
                    enc_outputs.append(enc_output)
                    
                bottleneck_outputs_stack = torch.stack(bottleneck_outputs)
                bottlenecks_tri_mean = torch.mean(bottleneck_outputs_stack, dim=0)
                bottlenecks_vslttxt_mean = torch.mean(torch.stack([bottleneck_outputs_stack[0,:,:,:], bottleneck_outputs_stack[2,:,:,:]]), dim=0)
                bottlenecks_vsltimg_mean = torch.mean(bottleneck_outputs_stack[:2,:,:,:], dim=0)
                all_bottleneck_stack = torch.stack([bottlenecks_tri_mean, bottlenecks_vsltimg_mean, bottlenecks_vslttxt_mean, bottleneck_outputs_stack[0,:,:,:]])
                
                # print("missing: ", missing)
                # print("missing: ", missing.shape)
                # print("all_bottleneck_stack: ", all_bottleneck_stack.shape)
                # print("self.idx_order: ", self.idx_order)
                # print("self.idx_order: ", self.idx_order.shape)
                
                bottlenecks = all_bottleneck_stack[missing, self.idx_order, :, :]
                if self.resbottle:
                    bottlenecks += res_bottles
                
                # bottlenecks = torch.where(missing[1].unsqueeze(1).unsqueeze(1) == 0, bottleneck_outputs[0], bottlenecks_mean)
                # bottlenecks = torch.where(varying_lengths[1].unsqueeze(1).unsqueeze(1) == 0, bottleneck_outputs[0], bottlenecks_mean)
                
        return enc_outputs, 0
    
class MBTEncoder(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
            n_modality: int,
            bottlenecks_n: int,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: list = [True, True],
            mask: list = [True, True]):
        super(MBTEncoder, self).__init__()

        self.n_modality = n_modality
        self.bottlenecks_n = bottlenecks_n
        self.fusion_startIdx = fusion_startidx

        self.use_pe = use_pe
        self.mask = mask
                
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        self.bottlenecks = nn.Parameter(torch.randn(1, self.bottlenecks_n, d_model))

        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, padded_inputs, lengths = None, return_attns = False):
        enc_slf_attn_list = []

        cls_token_per_modality = [cls_token.repeat(padded_inputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks = self.bottlenecks.repeat(padded_inputs[0].size(0), 1, 1)
        
        padded_inputs = [torch.cat([cls_token_per_modality[idx], padded_input], axis=1) for idx, padded_input in enumerate(padded_inputs)]
            
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        if self.n_modality == 3:#####
            self.mask=[True, True, True]
        for i in range(self.n_modality):
            if self.mask[i]:
                self_attn_masks.append(get_attn_pad_mask(padded_inputs[i], lengths[i]+1, padded_inputs[i].size(1)))
            else:
                self_attn_masks.append(None)
        
        if self.n_modality == 3:#####
            self.use_pe=[True, True, True]
        enc_inputs = []
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](padded_inputs[idx]) +
                    self.positional_encoding(padded_inputs[idx].size(1))
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](padded_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            if idx < self.fusion_startIdx:
                for modal_idx, enc_layer in enumerate(enc_layers):
                    enc_output, _ = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx])
                    enc_outputs.append(enc_output)      
                    
            else:
                bottleneck_outputs = []
                for modal_idx, enc_layer in enumerate(enc_layers):
                    b_enc_output = torch.cat([enc_inputs[modal_idx], bottlenecks], axis=1)
                    if len(bottleneck_self_attn_masks) < self.n_modality:
                        if self.mask[i]:
                            bottleneck_self_attn_masks.append(get_attn_pad_mask(b_enc_output, lengths[modal_idx]+1+self.bottlenecks_n, b_enc_output.size(1)))
                        else:
                            bottleneck_self_attn_masks.append(None)
                            
                    enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                    
                    bottleneck_outputs.append(enc_output[:, enc_inputs[modal_idx].size(1):, :])
                    enc_output = enc_output[:, :enc_inputs[modal_idx].size(1), :]
                    enc_outputs.append(enc_output)
                    
                bottleneck_outputs_stack = torch.stack(bottleneck_outputs)
                bottlenecks = torch.mean(bottleneck_outputs_stack, dim=0)
                
        return enc_outputs, 0