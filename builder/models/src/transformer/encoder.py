import torch.nn as nn
from torch import Tensor
from typing import Tuple
from builder.models.src.transformer.attention import MultiHeadAttention
from builder.models.src.transformer.module import PositionalEncoding, FeedForward, LayerNorm, FeedForwardUseConv
from builder.models.src.transformer.utils import *

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,             # dimension of transformer model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention_prenorm = LayerNorm(d_model)
        self.feed_forward_prenorm = LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.feed_forward = FeedForwardUseConv(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn

class TransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, 
                 d_input: int, 
                 n_layers: int, 
                 n_head: int,
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1, 
                 pe_maxlen: int = 5000, 
                 use_pe: bool = True, 
                 classification: bool = False, 
                 mask: bool = True):
        super(TransformerEncoder, self).__init__()
        # parameters
        self.use_pe = use_pe
        self.input_linear = False
        self.mask = mask
        self.classification = classification
        
        if d_input != d_model:
            # use linear transformation with layer norm to replace input embedding
            self.linear_in = nn.Linear(d_input, d_model)
            self.input_linear = True
        if self.classification:
            self.cls_tokens = nn.Parameter(torch.zeros(1, 1, d_model))
            # self.cls_tokens = nn.Parameter(torch.zeros(1, 1, d_model)).to(self.args.device)
            
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=n_head,
                d_ff=d_ff,
                dropout_p=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, padded_input, time_padded_input=None, input_lengths=None, return_attns=False):
        enc_slf_attn_list = []

        # Prepare masks
        # non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        if self.classification:
            cls_tokens = self.cls_tokens.repeat(padded_input.size(0), 1, 1)
            padded_input = torch.cat([cls_tokens, padded_input], axis=1)
        
        if self.mask:
            input_lengths += 1
            if time_padded_input is None:
                slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, padded_input.size(1))
        else:
            slf_attn_mask = None

        # Forward
        if self.input_linear:
            padded_input = self.linear_in(padded_input)
        
        if self.use_pe:
            enc_output = self.dropout(
                self.layer_norm_in(padded_input) +
                self.positional_encoding(padded_input.size(1)))

        else:
            if time_padded_input is None:
                enc_output = self.dropout(
                    self.layer_norm_in(padded_input))
            else:
                enc_output = self.dropout(torch.cat([padded_input, time_padded_input + self.positional_encoding(time_padded_input.size(1))], 1))
                slf_attn_mask = get_attn_pad_mask(enc_output, input_lengths, enc_output.size(1))
            
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0

class TrimodalTransformerEncoder_MT(nn.Module):
    """
    Trimodal Trnasformer Encoder with missing modality
    Designed by Destin 2023/03/04
    """
    
    def __init__(self,
            batch_size: int,
            n_modality: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            fusion_startidx: int = 0,
            multitoken: int = 0,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            txt_idx: int = 2,
            use_pe: list = [True, True, True],
            mask: list = [True, False, True]):
        super(TrimodalTransformerEncoder_MT, self).__init__()
        print("multitoken: ", multitoken)
        self.multitoken = multitoken
        self.use_pe = use_pe
        self.n_modality = n_modality
        self.fusion_idx = fusion_startidx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.mask = mask
        self.idx_order = torch.range(0, batch_size-1).type(torch.LongTensor)
        
        # CLASSIFICATION TOKENS
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        if self.multitoken == 1:
            self.final_cls_tokens = nn.Parameter(torch.zeros(1,3,d_model)).cuda()
            self.final_token_mask = torch.ones([3, 1]).cuda()
            self.specific_masks = torch.zeros(207,207).cuda()               # 3 + 25 + 50 + 129 = 207
            self.specific_masks[3,28:] = 1
            self.specific_masks[28:,3] = 1
            self.specific_masks[28,3:28] = 1
            self.specific_masks[3:28,28] = 1
            self.specific_masks[78,3:78] = 1
            self.specific_masks[3:78,78] = 1
            self.specific_masks[28,78:] = 1
            self.specific_masks[78:,28] = 1
            
            self.specific_masks[1,78:] = 1
            self.specific_masks[78:,1] = 1
            self.specific_masks[2,28:78] = 1
            self.specific_masks[28:78,2] = 1
            
            self.specific_masks[1,2] = 1
            self.specific_masks[2,1] = 1
            self.specific_masks[0,1:3] = 1
            self.specific_masks[1:3,0] = 1
            self.specific_masks[3,:3] = 1
            self.specific_masks[:3,3] = 1
            self.specific_masks = self.specific_masks.ge(0.5)

        else: 
            self.final_cls_tokens = nn.Parameter(torch.zeros(1,1,d_model)).cuda() 
            self.final_token_mask = torch.ones([1, 1]).cuda()
            self.specific_masks = torch.zeros(205,205).cuda()               # 1 + 25 + 50 + 129 = 205
            self.specific_masks[1,26:] = 1
            self.specific_masks[26:,1] = 1
            self.specific_masks[26,1:26] = 1
            self.specific_masks[1:26,26] = 1
            self.specific_masks[76,1:76] = 1
            self.specific_masks[1:76,76] = 1
            self.specific_masks[26,76:] = 1
            self.specific_masks[76:,26] = 1
            self.specific_masks = self.specific_masks.ge(0.5)
        
        self.layer_norms_in =  nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.specific_layer_stack =  nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(
                    d_model = d_model,
                    num_heads = n_head,
                    d_ff = d_ff,
                    dropout_p = dropout
                )for _ in range(n_layers)]) 
            for _ in range(n_modality)
        ])
        self.fusion_layer_stack =  nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_layers)
        ])
      
    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None, missing = None):
        enc_slf_attn_list = []
        # print("varying_lengths: ", varying_lengths)
        final_cls_tokens = self.final_cls_tokens.repeat(enc_outputs[0].size(0), 1, 1) 
        cls_token_per_modality = [cls_token.repeat(enc_outputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        final_cls_mask = self.final_token_mask.repeat(enc_outputs[0].size(0), 1, 1) 
        each_mask_stack = []
        enc_stack = []
        for n_modal in range(self.n_modality):
            transformer_in = torch.cat([cls_token_per_modality[n_modal], enc_outputs[n_modal]], 1)
            if self.use_pe:
                transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in) + self.positional_encoding(transformer_in.size(1)))
            else:
                transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in))
                
            enc_stack.append(transformer_in)
            varying_lengths[n_modal] += 1
            fixed_lengths[n_modal] += 1
            if n_modal == self.txt_idx:
                varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
            
            if self.mask[n_modal]:
                slf_attn_mask = get_attn_pad_mask(transformer_in[:,:fixed_lengths[n_modal],:], varying_lengths[n_modal], fixed_lengths[n_modal])
            else:
                slf_attn_mask = None
            each_mask_stack.append(slf_attn_mask)
                
        fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths, additional_cls_mask=final_cls_mask)
        # print("fusion_mask_stack: ", fusion_mask_stack.shape)
        # fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths)
        
        specific_masks = self.specific_masks.repeat(enc_outputs[0].size(0), 1, 1) 
        fusion_mask_stack = fusion_mask_stack + specific_masks
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,0,:])
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,1,:])
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,2,:])
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,3,:])
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.pcolormesh(fusion_mask_stack[5].detach().cpu())
        # plt.subplot(2,1,2)
        # plt.pcolormesh(specific_masks[5].detach().cpu())
        # plt.show()
        # exit(1)
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx

        fusion_first = True
        for block_idx in range(self.n_layers):
            if block_idx < self.fusion_idx:
                for n_modal, each_modal_layers in enumerate(self.specific_layer_stack):
                    enc_stack[n_modal], enc_slf_attn = each_modal_layers[block_idx](enc_stack[n_modal], each_mask_stack[n_modal])
            else:            
                if fusion_first:
                    enc_stack.insert(0, final_cls_tokens)
                    enc_output = torch.cat(enc_stack, 1)
                    fusion_first = False
                enc_output, enc_slf_attn = self.fusion_layer_stack[block_idx](enc_output, fusion_mask_stack)
                
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        ### for late fusion
        # if fusion_first:
        #     enc_output = torch.cat(enc_stack, 1)
        #     enc_output = torch.stack([enc_output[:,0,:], enc_output[:,25,:]])
        #     enc_output = torch.mean(enc_output, dim=0)
        #     enc_output = self.layer_norms_after_concat(self.d_model)

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0
        

class MultimodalTransformerEncoder(nn.Module):
    """
    Encoder of Transformer with Masking Changes allowing for Masking over more than one modality
    Designed by Destin 2023/01/16
    """
    def __init__(self,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            n_modality: int,
            fusion_idx: int = 0,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: bool = True,
            classification: bool = False,
            txt_idx: int = 1):
        super(MultimodalTransformerEncoder, self).__init__()

        self.use_pe = use_pe
        self.n_modality = n_modality
        self.classification = classification
        self.fusion_idx = fusion_idx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
        
        if self.classification:
            self.final_cls_tokens = nn.Parameter(torch.zeros(1,1,d_model)).cuda()
            self.cls_tokens = nn.Parameter(torch.zeros(1,n_modality,d_model)).cuda()
            self.final_token_mask = torch.ones([1, 1]).cuda()
            
            self.specific_masks = torch.zeros(155,155).cuda()
            self.specific_masks[1,26:] = 1
            self.specific_masks[26:,1] = 1
            self.specific_masks[26,2:26] = 1
            self.specific_masks[2:26,26] = 1
            self.specific_masks = self.specific_masks.ge(0.5)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.pcolormesh(self.specific_masks.detach().cpu())
            # plt.show()
        
        self.layer_norms_in =  nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.specific_layer_stack =  nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(
                    d_model = d_model,
                    num_heads = n_head,
                    d_ff = d_ff,
                    dropout_p = dropout
                )for _ in range(n_layers)]) 
            for _ in range(n_modality)
        ])
        self.fusion_layer_stack =  nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_layers)
        ])
      
    # fixed_lengths denotes the maximum lengths of the first modalities input
    # varying_lengths denotes the actual lengths of the inputs of the first modality
    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None):
        enc_slf_attn_list = []
        enc_stack = []
        each_mask_stack = []
        fusion_mask_stack = None
        # print("varying_lengths: ", varying_lengths)
        if self.classification:
            final_cls_tokens = self.final_cls_tokens.repeat(enc_outputs[0].size(0), 1, 1) 
            final_cls_mask = self.final_token_mask.repeat(enc_outputs[0].size(0), 1, 1) 
            cls_tokens = self.cls_tokens.repeat(enc_outputs[0].size(0), 1, 1)
            for n_modal in range(self.n_modality):
                transformer_in = torch.cat([cls_tokens[:,n_modal,:].unsqueeze(1), enc_outputs[n_modal]], 1)
                if self.use_pe:
                    transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in) + self.positional_encoding(transformer_in.size(1)))
                else:
                    transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in))
                enc_stack.append(transformer_in)
                varying_lengths[n_modal] += 1
                fixed_lengths[n_modal] += 1
                if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
                
                # mask generation
                # multi-token mask, specific token mask필요함
                slf_attn_mask = get_attn_pad_mask(transformer_in[:,:fixed_lengths[n_modal],:], varying_lengths[n_modal], fixed_lengths[n_modal])
                each_mask_stack.append(slf_attn_mask)
                
        fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths, additional_cls_mask=final_cls_mask)
        # fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths)
        
        specific_masks = self.specific_masks.repeat(enc_outputs[0].size(0), 1, 1) 
        # print("1 vslt each_mask_stack[0]: ", each_mask_stack[0][0,0,:])
        # print("1 vslt each_mask_stack[1]: ", each_mask_stack[0][1,0,:])
        # print("1 txt each_mask_stack[0]: ", each_mask_stack[1][0,0,:])
        # print("1 txt each_mask_stack[1]: ", each_mask_stack[1][1,0,:])
        # print("1 fusion_mask_stack[0]: ", fusion_mask_stack[0,0,:])
        # print("1 fusion_mask_stack[1]: ", fusion_mask_stack[1,0,:])
        
        fusion_mask_stack = fusion_mask_stack + specific_masks
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,0,:])
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,1,:])
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,2,:])
        # print("2 fusion_mask_stack[1]: ", fusion_mask_stack[5,3,:])
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.pcolormesh(fusion_mask_stack[3].detach().cpu())
        # plt.subplot(2,1,2)
        # plt.pcolormesh(specific_masks[3].detach().cpu())
        # plt.show()
        # exit(1)
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx

        fusion_first = True
        for block_idx in range(self.n_layers):
            if block_idx < self.fusion_idx:
                for n_modal, each_modal_layers in enumerate(self.specific_layer_stack):
                    enc_stack[n_modal], enc_slf_attn = each_modal_layers[block_idx](enc_stack[n_modal], each_mask_stack[n_modal])
            else:            
                if fusion_first:
                    enc_stack.insert(0, final_cls_tokens)
                    enc_output = torch.cat(enc_stack, 1)
                    fusion_first = False
                enc_output, enc_slf_attn = self.fusion_layer_stack[block_idx](enc_output, fusion_mask_stack)
                
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if fusion_first:
            enc_output = torch.cat(enc_stack, 1)
            enc_output = torch.stack([enc_output[:,0,:], enc_output[:,25,:]])
            enc_output = torch.mean(enc_output, dim=0)
            enc_output = self.layer_norms_after_concat(self.d_model)

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0
        
class MultiToken_MultimodalTransformerEncoder(nn.Module):
    """
    Encoder of Transformer with Masking Changes allowing for Masking over more than one modality
    Designed by Destin 2023/01/16
    """
    def __init__(self,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            n_modality: int,
            fusion_idx: int = 0,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: bool = True,
            classification: bool = False,
            txt_idx: int = 1):
        super(MultiToken_MultimodalTransformerEncoder, self).__init__()

        self.use_pe = use_pe
        self.n_modality = n_modality
        self.classification = classification
        self.fusion_idx = fusion_idx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
        
        if self.classification:
            self.final_cls_tokens = nn.Parameter(torch.zeros(1,1,d_model)).cuda()
            self.cls_tokens = nn.Parameter(torch.zeros(1,n_modality,d_model)).cuda()
            self.final_token_mask = torch.ones([1, 1]).cuda()
            
            self.specific_masks = torch.zeros(155,155).cuda()
            self.specific_masks[1,26:] = 1
            self.specific_masks[1,0] = 1
            self.specific_masks[26:,1] = 1
            self.specific_masks[26,0:26] = 1
            self.specific_masks[2:26,26] = 1
            self.specific_masks = self.specific_masks.ge(0.5)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.pcolormesh(self.specific_masks.detach().cpu())
            # plt.show()
            # exit(1)
        
        self.layer_norms_in =  nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.specific_layer_stack =  nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(
                    d_model = d_model,
                    num_heads = n_head,
                    d_ff = d_ff,
                    dropout_p = dropout
                )for _ in range(n_layers)]) 
            for _ in range(n_modality)
        ])
        self.fusion_layer_stack =  nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_layers)
        ])
      
    # fixed_lengths denotes the maximum lengths of the first modalities input
    # varying_lengths denotes the actual lengths of the inputs of the first modality
    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None):
        enc_slf_attn_list = []
        enc_stack = []
        each_mask_stack = []
        fusion_mask_stack = None
        # print("varying_lengths: ", varying_lengths)
        if self.classification:
            final_cls_tokens = self.final_cls_tokens.repeat(enc_outputs[0].size(0), 1, 1) 
            final_cls_mask = self.final_token_mask.repeat(enc_outputs[0].size(0), 1, 1) 
            cls_tokens = self.cls_tokens.repeat(enc_outputs[0].size(0), 1, 1)
            for n_modal in range(self.n_modality):
                transformer_in = torch.cat([cls_tokens[:,n_modal,:].unsqueeze(1), enc_outputs[n_modal]], 1)
                if self.use_pe:
                    transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in) + self.positional_encoding(transformer_in.size(1)))
                else:
                    transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in))
                enc_stack.append(transformer_in)
                varying_lengths[n_modal] += 1
                fixed_lengths[n_modal] += 1
                if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
                
                # mask generation
                # multi-token mask, specific token mask필요함
                slf_attn_mask = get_attn_pad_mask(transformer_in[:,:fixed_lengths[n_modal],:], varying_lengths[n_modal], fixed_lengths[n_modal])
                each_mask_stack.append(slf_attn_mask)
                
        fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths, additional_cls_mask=final_cls_mask)
        # fusion_mask_stack = get_multi_attn_pad_mask(final_cls_mask, enc_stack, varying_lengths, fixed_lengths)
        
        specific_masks = self.specific_masks.repeat(enc_outputs[0].size(0), 1, 1) 
        fusion_mask_stack = fusion_mask_stack + specific_masks

        if fusion_idx is not None:
            self.fusion_idx = fusion_idx

        fusion_first = True
        for block_idx in range(self.n_layers):
            if block_idx < self.fusion_idx:
                for n_modal, each_modal_layers in enumerate(self.specific_layer_stack):
                    enc_stack[n_modal], enc_slf_attn = each_modal_layers[block_idx](enc_stack[n_modal], each_mask_stack[n_modal])
            else:            
                if fusion_first:
                    enc_stack.insert(0, final_cls_tokens)
                    enc_output = torch.cat(enc_stack, 1)
                    fusion_first = False
                enc_output, enc_slf_attn = self.fusion_layer_stack[block_idx](enc_output, fusion_mask_stack)
                
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if fusion_first:
            enc_output = torch.cat(enc_stack, 1)
            enc_output = torch.stack([enc_output[:,0,:], enc_output[:,25,:]])
            enc_output = torch.mean(enc_output, dim=0)
            enc_output = self.layer_norms_after_concat(self.d_model)

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0
        
class MultiToken_MultimodalTransformerEncoder_sym(nn.Module):
    """
    Encoder of Transformer with Masking Changes allowing for Masking over more than one modality
    Designed by Destin 2023/01/16
    """
    def __init__(self,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            n_modality: int,
            fusion_idx: int = 0,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: bool = True,
            classification: bool = False,
            txt_idx: int = 1):
        super(MultiToken_MultimodalTransformerEncoder_sym, self).__init__()

        self.use_pe = use_pe
        self.n_modality = n_modality
        self.classification = classification
        self.fusion_idx = fusion_idx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
        
        if self.classification:
            self.final_cls_tokens = nn.Parameter(torch.zeros(1,1,d_model)).cuda()
            self.cls_tokens = nn.Parameter(torch.zeros(1,n_modality,d_model)).cuda()
            self.final_token_mask = torch.ones([1, 1]).cuda()
            
            self.specific_masks = torch.zeros(155,155).cuda()
            
            self.specific_masks[1,0] = 1
            self.specific_masks[0,1] = 1
            
            self.specific_masks[1,26:] = 1
            self.specific_masks[26:,1] = 1
            
            self.specific_masks[26,0:26] = 1
            self.specific_masks[0:26,26] = 1
            self.specific_masks = self.specific_masks.ge(0.5)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.pcolormesh(self.specific_masks.detach().cpu())
            # plt.show()
            # exit(1)
        
        self.layer_norms_in =  nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.specific_layer_stack =  nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(
                    d_model = d_model,
                    num_heads = n_head,
                    d_ff = d_ff,
                    dropout_p = dropout
                )for _ in range(n_layers)]) 
            for _ in range(n_modality)
        ])
        self.fusion_layer_stack =  nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_layers)
        ])
      
    # fixed_lengths denotes the maximum lengths of the first modalities input
    # varying_lengths denotes the actual lengths of the inputs of the first modality
    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None):
        enc_slf_attn_list = []
        enc_stack = []
        each_mask_stack = []
        fusion_mask_stack = None
        # print("varying_lengths: ", varying_lengths)
        if self.classification:
            final_cls_tokens = self.final_cls_tokens.repeat(enc_outputs[0].size(0), 1, 1) 
            final_cls_mask = self.final_token_mask.repeat(enc_outputs[0].size(0), 1, 1) 
            cls_tokens = self.cls_tokens.repeat(enc_outputs[0].size(0), 1, 1)
            for n_modal in range(self.n_modality):
                transformer_in = torch.cat([cls_tokens[:,n_modal,:].unsqueeze(1), enc_outputs[n_modal]], 1)
                if self.use_pe:
                    transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in) + self.positional_encoding(transformer_in.size(1)))
                else:
                    transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in))
                enc_stack.append(transformer_in)
                varying_lengths[n_modal] += 1
                fixed_lengths[n_modal] += 1
                if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
                
                # mask generation
                # multi-token mask, specific token mask필요함
                slf_attn_mask = get_attn_pad_mask(transformer_in[:,:fixed_lengths[n_modal],:], varying_lengths[n_modal], fixed_lengths[n_modal])
                each_mask_stack.append(slf_attn_mask)
                
        fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths, additional_cls_mask=final_cls_mask)
        # fusion_mask_stack = get_multi_attn_pad_mask(final_cls_mask, enc_stack, varying_lengths, fixed_lengths)
        
        specific_masks = self.specific_masks.repeat(enc_outputs[0].size(0), 1, 1) 
        fusion_mask_stack = fusion_mask_stack + specific_masks

        if fusion_idx is not None:
            self.fusion_idx = fusion_idx

        fusion_first = True
        for block_idx in range(self.n_layers):
            if block_idx < self.fusion_idx:
                for n_modal, each_modal_layers in enumerate(self.specific_layer_stack):
                    enc_stack[n_modal], enc_slf_attn = each_modal_layers[block_idx](enc_stack[n_modal], each_mask_stack[n_modal])
            else:            
                if fusion_first:
                    enc_stack.insert(0, final_cls_tokens)
                    enc_output = torch.cat(enc_stack, 1)
                    fusion_first = False
                enc_output, enc_slf_attn = self.fusion_layer_stack[block_idx](enc_output, fusion_mask_stack)
                
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if fusion_first:
            enc_output = torch.cat(enc_stack, 1)
            enc_output = torch.stack([enc_output[:,0,:], enc_output[:,25,:]])
            enc_output = torch.mean(enc_output, dim=0)
            enc_output = self.layer_norms_after_concat(self.d_model)

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0
 
class CrossmodalTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,             # dimension of transformer model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ) -> None:
        super(CrossmodalTransformerEncoderLayer, self).__init__()
        self.attention_prenorm_q = LayerNorm(d_model)
        self.attention_prenorm_kv = LayerNorm(d_model)
        self.feed_forward_prenorm = LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.feed_forward = FeedForwardUseConv(d_model, d_ff, dropout_p)

    def forward(self, q_inputs: Tensor, kv_inputs: Tensor, self_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        residual = q_inputs
        inputs_q = self.attention_prenorm_q(q_inputs)
        inputs_kv = self.attention_prenorm_kv(kv_inputs)
        outputs, attn = self.self_attention(inputs_q, inputs_kv, inputs_kv, self_attn_mask) #self_attn_mask should be based on k, v matrices
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn

class BimodalTransformerEncoder(nn.Module):
    """
    Encoder of Transformer with Masking Changes allowing for Masking over two types of modalities
    """

    def __init__(self,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: bool = True,
            classification: bool = False,
            mask: bool = True):
        super(BimodalTransformerEncoder, self).__init__()

        self.use_pe = use_pe
        self.input_linear = False
        self.mask = mask
        self.classification = classification

        if d_input != d_model:
            self.linear_in = nn.Linear(d_input, d_model)
            self.input_linear = True
        if self.classification:
            self.cls_tokens = nn.Parameter(torch.zeros(1,1,d_model))
        
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack =  nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_layers)
        ])

    # first_size denotes the maximum lengths of the first modalities input
    # first_lengths denotes the actual lengths of the inputs of the first modality
    def forward(self, padded_input, first_size = None, first_lengths = None, second_lengths = None, return_attns = False):
        enc_slf_attn_list = []

        print("first_size: ", first_size)
        print("first_lengths: ", first_lengths)
        print("second_lengths: ", second_lengths)

        if self.classification:
            cls_tokens = self.cls_tokens.repeat(padded_input.size(0), 1, 1)
            padded_input = torch.cat([cls_tokens, padded_input], axis=1)
        
        if self.mask:
            if self.classification:
                first_size += 1
            print("expand_length; ", first_size)
            print("first_lengths; ", first_lengths)
            print("second_lengths; ", second_lengths)
            first_attn_mask = get_attn_pad_mask(padded_input[:,:first_size,:], first_lengths, first_size)
            second_attn_mask = get_attn_pad_mask(padded_input[:,first_size:,:], second_lengths, padded_input.size(1) - first_size)

            total_length = padded_input.size(1)
            first_multiplier = total_length // first_attn_mask.size(1) + 1
            second_multiplier = total_length // second_attn_mask.size(1) + 1

            first_attn_mask = first_attn_mask.repeat(1, first_multiplier, 1)[:,:total_length,:]
            second_attn_mask = second_attn_mask.repeat(1, second_multiplier, 1)[:,:total_length,:]

            print("first_attn_mask: ", first_attn_mask.shape)
            print("second_attn_mask: ", second_attn_mask.shape)
            print("second_attn_mask: ", second_attn_mask[0])
            print("padded_input: ", padded_input.shape)

            slf_attn_mask = torch.cat([first_attn_mask, second_attn_mask], dim=2)
        else:
            slf_attn_mask = None
        
        if self.input_linear:
            padded_input = self.linear_in(padded_input)
        
        if self.use_pe:
            enc_output = self.dropout(
                self.layer_norm_in(padded_input) +
                self.positional_encoding(padded_input.size(1))
            )
        else:
            enc_output = self.dropout(
                self.layer_norm_in(padded_input)
            )
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0

class TrimodalTransformerEncoder(nn.Module):
    """
    Encoder of Transformer with Masking Changes allowing for Masking over two types of modalities
    """

    def __init__(self,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: bool = True,
            classification: bool = False,
            mask: bool = True):
        super(TrimodalTransformerEncoder, self).__init__()

        self.use_pe = use_pe
        self.input_linear = False
        self.mask = mask
        self.classification = classification

        if d_input != d_model:
            self.linear_in = nn.Linear(d_input, d_model)
            self.input_linear = True
        if self.classification:
            self.cls_tokens = nn.Parameter(torch.zeros(1,1,d_model))
        
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack =  nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_layers)
        ])

    # first_size denotes the maximum lengths of the first modalities input
    # first_lengths denotes the actual lengths of the inputs of the first modality
    def forward(self, padded_input, first_size = None, first_lengths = None, second_size = None, second_lengths = None, third_lengths = None, return_attns = False):
        enc_slf_attn_list = []

        # print("first_size: ", first_size)
        # print("first_lengths: ", first_lengths)
        # print("second_lengths: ", second_lengths)

        if self.classification:
            cls_tokens = self.cls_tokens.repeat(padded_input.size(0), 1, 1)
            padded_input = torch.cat([cls_tokens, padded_input], axis=1)
        
        if self.mask:
            if self.classification:
                first_size += 1
                second_size += 1#### 맞을까? 
            # print("expand_length; ", first_size)
            # print("first_lengths; ", first_lengths)
            # print("second_lengths; ", second_lengths)
            first_attn_mask = get_attn_pad_mask(padded_input[:,:first_size,:], first_lengths, first_size)
            second_attn_mask = get_attn_pad_mask(padded_input[:,first_size:second_size,:], second_lengths, second_size)
            third_attn_mask = get_attn_pad_mask(padded_input[:,second_size:,:], third_lengths, padded_input.size(1) - first_size - second_size)

            total_length = padded_input.size(1)
            first_multiplier = total_length // first_attn_mask.size(1) + 1
            second_multiplier = total_length // second_attn_mask.size(1) + 1
            third_multiplier = total_length // third_attn_mask.size(1) + 1

            first_attn_mask = first_attn_mask.repeat(1, first_multiplier, 1)[:,:total_length,:]
            second_attn_mask = second_attn_mask.repeat(1, second_multiplier, 1)[:,:total_length,:]
            third_attn_mask = third_attn_mask.repeat(1, third_multiplier, 1)[:,:total_length,:]

            # print("first_attn_mask: ", first_attn_mask.shape)
            # print("second_attn_mask: ", second_attn_mask.shape)
            # print("second_attn_mask: ", second_attn_mask[0])
            # print("padded_input: ", padded_input.shape)

            slf_attn_mask = torch.cat([first_attn_mask, second_attn_mask, third_attn_mask], dim=2)
        else:
            slf_attn_mask = None
        
        if self.input_linear:
            padded_input = self.linear_in(padded_input)
        
        if self.use_pe:
            enc_output = self.dropout(
                self.layer_norm_in(padded_input) +
                self.positional_encoding(padded_input.size(1))
            )
        else:
            enc_output = self.dropout(
                self.layer_norm_in(padded_input)
            )
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0
        
class CrossTransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, 
                 d_input: int, 
                 n_layers: int, 
                 n_head: int,
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1, 
                 pe_maxlen: int = 5000, 
                 use_pe: bool = True, 
                 classification: bool = False, 
                 mask: bool = True):
        super(CrossTransformerEncoder, self).__init__()
        # parameters
        self.use_pe = use_pe
        self.input_linear = False
        self.mask = mask
        self.classification = classification
        
        if d_input != d_model:
            # use linear transformation with layer norm to replace input embedding
            self.linear_in = nn.Linear(d_input, d_model)
            self.input_linear = True
        if self.classification:
            self.cls_tokens = nn.Parameter(torch.zeros(1, 1, d_model))
            
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_stack = nn.ModuleList([
            CrossmodalTransformerEncoderLayer(
                d_model=d_model,
                num_heads=n_head,
                d_ff=d_ff,
                dropout_p=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, padded_q_input, padded_kv_input, q_input_lengths=None, kv_input_lengths=None, return_attns=False):
        enc_slf_attn_list = []

        # Prepare masks
        # non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        if self.classification:
            cls_tokens = self.cls_tokens.repeat(padded_q_input.size(0), 1, 1)
            padded_q_input = torch.cat([cls_tokens, padded_q_input], axis=1)
        
        if self.mask:
            slf_attn_mask = get_cross_attn_pad_mask(padded_kv_input, kv_input_lengths, q_input_lengths)
            # print("slf_attn_mask: ", slf_attn_mask.shape)
            # print("slf_attn_mask: ", slf_attn_mask[0,0,:])
            # print("slf_attn_mask: ", slf_attn_mask[1,0,:])
            # print("slf_attn_mask: ", slf_attn_mask[2,0,:])
        else:
            slf_attn_mask = None
        # Forward
        if self.input_linear:
            padded_q_input = self.linear_in(padded_q_input)
        
        if self.use_pe:
            enc_output = self.dropout(
                self.layer_norm_in(padded_q_input) +
                self.positional_encoding(padded_q_input.size(1)))
        else:
            enc_output = self.dropout(
                self.layer_norm_in(padded_q_input))
            
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(q_inputs = enc_output, 
                                                 kv_inputs = padded_kv_input, 
                                                 self_attn_mask = slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0
             
class SelfCrossTransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, 
                 d_input: int, 
                 n_layers: int, 
                 n_head: int,
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1, 
                 pe_maxlen: int = 5000, 
                 use_pe: bool = True, 
                 classification: bool = False, 
                 mask: bool = True):
        super(SelfCrossTransformerEncoder, self).__init__()
        # parameters
        self.use_pe = use_pe
        self.input_linear = False
        self.mask = mask
        self.classification = classification
        
        if d_input != d_model:
            # use linear transformation with layer norm to replace input embedding
            self.linear_in = nn.Linear(d_input, d_model)
            self.input_linear = True
        if self.classification:
            self.cls_tokens = nn.Parameter(torch.zeros(1, 1, d_model))
            
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_stack = nn.ModuleList([])
        for _ in range(n_layers):
            self.layer_stack.append(nn.ModuleList([
                TransformerEncoderLayer(d_model=d_model, num_heads=n_head, d_ff=d_ff, dropout_p=dropout),
                CrossmodalTransformerEncoderLayer(d_model=d_model, num_heads=n_head, d_ff=d_ff, dropout_p=dropout)
                ]))

    def forward(self, padded_q_input, padded_kv_input, q_input_lengths=None, q_input_length=None, kv_input_lengths=None, return_attns=False):
        enc_slf_attn_list = []

        # Prepare masks
        # non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        if self.classification:
            cls_tokens = self.cls_tokens.repeat(padded_q_input.size(0), 1, 1)
            padded_q_input = torch.cat([cls_tokens, padded_q_input], axis=1)
        # print("padded_kv_input: ", padded_kv_input.shape)
        # print("kv_input_lengths: ", kv_input_lengths)
        # print("q_input_lengths: ", q_input_lengths)
        # print("padded_q_input: ", padded_q_input.shape)
        # print("q_input_lengths: ", q_input_lengths)
        # print("padded_q_input: ", padded_q_input.size(1))
        if self.mask:
            cross_attn_mask = get_cross_attn_pad_mask(padded_kv_input, kv_input_lengths, q_input_length)
            slf_attn_mask = get_attn_pad_mask(padded_q_input, q_input_lengths+1, padded_q_input.size(1))
            # print("padded_input: ", padded_kv_input.shape)
            # print("padded_input: ", padded_kv_input[0,:,0])
            # print("cross_attn_mask: ", cross_attn_mask.shape)
            # print("cross_attn_mask: ", cross_attn_mask[0,0,:])
            # print("slf_attn_mask: ", slf_attn_mask.shape)
            # print("slf_attn_mask: ", slf_attn_mask[0,0,:])
        else:
            slf_attn_mask = None
        # Forward
        if self.input_linear:
            padded_q_input = self.linear_in(padded_q_input)
        
        if self.use_pe:
            enc_output = self.dropout(
                self.layer_norm_in(padded_q_input) +
                self.positional_encoding(padded_q_input.size(1)))
        else:
            enc_output = self.dropout(
                self.layer_norm_in(padded_q_input))
            
        for self_transformer, cross_transformer in self.layer_stack:
            self_enc_output, self_enc_slf_attn = self_transformer(enc_output, slf_attn_mask)
                        
            enc_output, cross_enc_slf_attn = cross_transformer(q_inputs = self_enc_output, 
                                                 kv_inputs = padded_kv_input, 
                                                 self_attn_mask = cross_attn_mask)
            if return_attns:
                enc_slf_attn_list += [self_enc_slf_attn]
                enc_slf_attn_list += [cross_enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 0