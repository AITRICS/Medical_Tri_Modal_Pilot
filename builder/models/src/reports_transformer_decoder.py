# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from control.config import args
from builder.models.src.transformer.attention import MultiHeadAttention
from builder.models.src.transformer.module import Linear, LayerNorm, Embedding, PositionalEncoding
from builder.models.src.transformer import BaseDecoder
from builder.models.src.transformer.module import FeedForwardUseConv
from builder.models.src.transformer.utils import (
    get_decoder_self_attn_mask,
    get_attn_pad_mask,
)


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_prenorm = LayerNorm(d_model)
        self.encoder_attention_prenorm = LayerNorm(d_model)
        self.feed_forward_prenorm = LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardUseConv(d_model, d_ff, dropout_p)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Optional[Tensor] = None,
            encoder_outputs_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.encoder_attention_prenorm(outputs)
        outputs, encoder_attn = self.encoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_outputs_mask) # query, key, value 
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, self_attn, encoder_attn


class TransformerDecoder(BaseDecoder):
    """
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        num_classes: umber of classes
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of decoder layers
        num_heads: number of attention heads
        dropout_p: probability of dropout
        pad_id: identification of pad token
        eos_id: identification of end of sentence token
    """

    def __init__(
            self,
            num_classes: int,               # number of classes
            d_model: int = 512,             # dimension of model
            d_ff: int = 512,                # dimension of feed forward network
            num_layers: int = 6,            # number of decoder layers
            num_heads: int = 8,             # number of attention heads
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
            sos_id: int = 1,                # identification of start of sentence token
            eos_id: int = 2,                # identification of end of sentence token
            max_length: int = 400,          # max length of decoding
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            LayerNorm(d_model),
            Linear(d_model, num_classes, bias=False),
        )

    def forward(self, targets_copy: Tensor, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tensor:
        
        batch_size = targets_copy.size(0)
        # # missing일 경우 뒤의 0을 제거, missing이 아닐 경우 self.eos_id 제거
        # targets = torch.empty(targets_copy.shape[0],targets_copy.shape[1]-1, dtype = torch.long).to(args.device)
        # for i in range (batch_size):
        #     if self.eos_id not in targets_copy[i]:
        #         targets[i] = targets_copy[i][:-1]
        #     else:
        #         targets[i] = targets_copy[i][targets_copy[i] != self.eos_id ]
                
        #targets = targets[targets != self.eos_id].view(batch_size, -1)
        # targets[i] = targets_copy[i][targets_copy[i] != self.eos_id ]
        targets_copy = targets_copy[:,:-1]
        targets = targets_copy.view(batch_size, -1)
        target_length = targets.size(1)
        
        # decoder_inputs=targets,
        # decoder_input_lengths=target_lengths,
        # encoder_outputs=encoder_outputs,
        # encoder_output_lengths=encoder_output_lengths,
        # positional_encoding_length=target_length,

        # dec_self_attn_pad_mask = get_attn_pad_mask(
        #     decoder_inputs, decoder_input_lengths, decoder_inputs.size(1)
        # )
        # dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        
        #seq_k -> context vector, seq_q-> target, pad_id 이지만 디코더의 마스크드 셀프 어텐션 : Query = Key = Value
        self_attn_mask = get_decoder_self_attn_mask(targets, targets, self.pad_id) #디코더의 마스크드 셀프 어텐션
        encoder_outputs_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, target_length)

        outputs = self.embedding(targets) + self.positional_encoding(target_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_outputs_mask=encoder_outputs_mask,
            )

        # predicted_log_probs = self.fc(outputs).log_softmax(dim=-1)
        no_predicted_log_probs = self.fc(outputs)
        return no_predicted_log_probs

    # @torch.no_grad()
    # def decode(self, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tensor:
    #     pass
    #     # batch_size = encoder_outputs.size(0)
        
    #     # predictions = encoder_outputs.new_zeros(batch_size, self.max_length).long()
    #     # predictions[:, 0] = self.sos_id

    #     # for di in range(1, self.max_length):
    #     #     step_outputs = self.forward(predictions, encoder_outputs, encoder_output_lengths)
    #     #     step_outputs = step_outputs.max(dim=-1, keepdim=False)[1]# [batch_size, 1023]
    #     #     predictions[:, di] = step_outputs[:, di-1]
    #     #     if di == 1023:
    #     #         print("end")

    #     # return step_outputs #predictions
    
  