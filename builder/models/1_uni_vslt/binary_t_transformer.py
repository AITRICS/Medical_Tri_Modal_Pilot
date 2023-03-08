import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
# from builder.models.src.transformer.module import PositionalEncoding
from builder.models.src.transformer import *

# def get_tgt_mask(self, size) -> torch.tensor:
#         # Generates a squeare matrix where the each row allows one word more to be seen
#         mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
#         mask = mask.float()
#         mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
#         mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
#         # EX for size=5:
#         # [[0., -inf, -inf, -inf, -inf],
#         #  [0.,   0., -inf, -inf, -inf],
#         #  [0.,   0.,   0., -inf, -inf],
#         #  [0.,   0.,   0.,   0., -inf],
#         #  [0.,   0.,   0.,   0.,   0.]]
        
#         return mask

def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)

class BINARY_T_TRANSFORMER(nn.Module):
        def __init__(self, args):
            super().__init__()      
            self.args = args
            self.num_nodes = len(args.vitalsign_labtest)
            self.t_len = args.window_size
            self.num_layers = args.transformer_num_layers
            self.n_head = args.transformer_num_head
            enc_model_dim = args.transformer_dim
            dropout = args.dropout
            
            activation = 'relu'
            self.activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['prelu', nn.PReLU()],
                    ['relu', nn.ReLU(inplace=True)],
                    ['tanh', nn.Tanh()],
                    ['sigmoid', nn.Sigmoid()],
                    ['leaky_relu', nn.LeakyReLU(0.2)],
                    ['elu', nn.ELU()]
            ])

            self.init_fc = nn.Linear(self.num_nodes, enc_model_dim)
            
            self.transformer_encoder = TransformerEncoder(
                                        d_input=enc_model_dim,
                                        n_layers=self.num_layers,
                                        n_head=self.n_head,
                                        d_model=enc_model_dim,
                                        d_ff=enc_model_dim*4,
                                        dropout=dropout,
                                        pe_maxlen=200,
                                        use_pe=True,
                                        classification=True)
            
            self.classifier = nn.Sequential(
            nn.Linear(in_features=enc_model_dim, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True))
    
            self.relu = self.activations[activation]
            self.sigmoid = self.activations['sigmoid']

        
        def forward(self, x, h, m, d, x_m, input_lengths):
            emb = self.init_fc(x)
            output, _ = self.transformer_encoder(emb, input_lengths=input_lengths+1)

            output = self.classifier(output[:,0,:])
            # exit(1)
            return self.sigmoid(output)




