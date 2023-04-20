import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Any

from builder.models.src.transformer.module import Linear


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int, scale: bool = True, block_mask: list = None) -> None:
        super(ScaledDotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1
        self.block_mask = block_mask

        if self.block_mask is not None:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Any] = None
    ) -> Tuple[Tensor, Tensor]:
        # os.system("echo \'query {}\'".format(str(query.shape)))
        # os.system("echo \'key {}\'".format(str(key.shape)))
        # print(" ")
        # score = ((torch.nan_to_num(torch.bmm(query, key.transpose(1, 2)), nan=0.0, posinf=1.0)) / self.sqrt_dim)  # + 2e-10
        score = ((torch.bmm(query, key.transpose(1, 2))) / self.sqrt_dim)  # + 2e-10
        # print("mask: ", mask.shape)
        # print("mask: ", mask[0])
        if mask is not None:
            score.masked_fill_(mask, -65504)
            # score.masked_fill_(mask, -1e9)
        
        # attn = torch.nan_to_num(F.softmax(score, -1), nan=0.0, posinf=1.0)
        attn = F.softmax(score, -1)

        if self.block_mask is not None:
            attn = attn * self.block_mask
            context = self.gamma * torch.bmm(attn, value)
            
        else:
            context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, block_mask: list = None) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        self.key_proj = Linear(dim, self.d_head * num_heads)
        self.value_proj = Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, scale=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        dim (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            dim: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "d_model % num_heads should be zero."

        self.dim = dim
        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(dim)

        self.query_proj = Linear(dim, dim)
        self.key_proj = Linear(dim, dim)
        self.value_proj = Linear(dim, dim)
        self.pos_proj = Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(dim, dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -65504)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.dim)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        dim (int): dimension of model
        attn_dim (int): dimension of attention
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, dim: int = 1024, attn_dim: int = 1024, smoothing: bool = False) -> None:
        super(LocationAwareAttention, self).__init__()
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_proj = Linear(dim, attn_dim, bias=False)
        self.value_proj = Linear(dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.fc = Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_alignment_energy: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length)

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(dim=1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)

        alignmment_energy = self.fc(torch.tanh(
                self.query_proj(query)
                + self.value_proj(value)
                + last_alignment_energy
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            alignmment_energy = torch.sigmoid(alignmment_energy)
            alignmment_energy = torch.div(alignmment_energy, alignmment_energy.sum(dim=-1).unsqueeze(dim=-1))

        else:
            alignmment_energy = F.softmax(alignmment_energy, dim=-1)

        context = torch.bmm(alignmment_energy.unsqueeze(dim=1), value)

        return context, alignmment_energy