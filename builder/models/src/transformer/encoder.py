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
            dropout: float = 0.1,
            pe_maxlen: int = 3000,
            txt_idx: int = 2,
            use_pe: list = [True, True, True],
            mask: list = [True, False, True]):
        super(TrimodalTransformerEncoder_MT, self).__init__()
        self.use_pe = use_pe
        self.n_modality = n_modality
        self.fusion_idx = fusion_startidx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.mask = mask
        self.idx_order = torch.range(0, batch_size-1).type(torch.LongTensor)
        
        # CLASSIFICATION TOKENS
        # self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        self.cls_token_for_img = nn.Parameter(torch.randn(1, 1, d_model))
    
        self.final_cls_tokens = nn.Parameter(torch.zeros(1,1,d_model)).cuda() 
        self.final_token_mask = torch.ones([1, 1]).cuda()
        # self.specific_masks = torch.zeros(205,205).cuda()
        # self.specific_masks[1,26:] = 1
        # self.specific_masks[26:,1] = 1
        # self.specific_masks = self.specific_masks.ge(0.5)
    
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
        final_cls_tokens = self.final_cls_tokens.repeat(enc_outputs[0].size(0), 1, 1) 
        cls_token_for_img = self.cls_token_for_img.repeat(enc_outputs[0].size(0), 1, 1)
        final_cls_mask = self.final_token_mask.repeat(enc_outputs[0].size(0), 1, 1) 
        each_mask_stack = []
        enc_stack = []
        for n_modal in range(self.n_modality):
            if n_modal == 1:
                transformer_in = torch.cat([cls_token_for_img, enc_outputs[n_modal]], 1)
                varying_lengths[n_modal] += 1
                fixed_lengths[n_modal] += 1
            else:
                transformer_in = enc_outputs[n_modal]
            if self.use_pe:
                transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in) + self.positional_encoding(transformer_in.size(1)))
            else:
                transformer_in = self.dropout(self.layer_norms_in[n_modal](transformer_in))
                
            enc_stack.append(transformer_in)
            if n_modal == self.txt_idx:
                varying_lengths[n_modal][varying_lengths[n_modal] == 2] = 0
            
            if self.mask[n_modal]:
                slf_attn_mask = get_attn_pad_mask(transformer_in[:,:fixed_lengths[n_modal],:], varying_lengths[n_modal], fixed_lengths[n_modal])
            else:
                slf_attn_mask = None
            each_mask_stack.append(slf_attn_mask)
        fusion_mask_stack = get_multi_attn_pad_mask(enc_stack, varying_lengths, fixed_lengths, additional_cls_mask=final_cls_mask)
        
        specific_masks = torch.zeros([fusion_mask_stack.size(1),fusion_mask_stack.size(2)])
        specific_masks[-178,:-178] = 1
        specific_masks[-178,-128:] = 1
        specific_masks = specific_masks.type(torch.bool).repeat(enc_outputs[0].size(0), 1, 1).cuda() 
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