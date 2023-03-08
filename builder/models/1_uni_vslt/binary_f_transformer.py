import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from builder.models.src.transformer.module import PositionalEncoding
from builder.models.src.transformer import *

class BINARY_F_TRANSFORMER(nn.Module):
        def __init__(self, args):
            super().__init__()      
            self.args = args
            self.num_nodes = len(args.vitalsign_labtest)
            self.t_len = args.window_size
            self.num_layers = args.transformer_num_layers
            enc_model_dim = args.transformer_dim
            self.n_head = args.transformer_num_head
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

            self.init_fc_list = nn.ModuleList()
            for _ in range(self.num_nodes):
                    self.init_fc_list.append(nn.Linear(self.t_len, enc_model_dim))
            self.init_ln = nn.LayerNorm(self.num_nodes)
            self.init_relu = nn.ReLU(inplace=True)
            
            self.transformer_encoder = TransformerEncoder(
                                        d_input=enc_model_dim,
                                        n_layers=self.num_layers,
                                        n_head=self.n_head,
                                        d_model=enc_model_dim,
                                        d_ff=enc_model_dim*4,
                                        dropout=dropout,
                                        pe_maxlen=25,
                                        use_pe=False,
                                        classification=True,
                                        mask=False)
            
            self.classifier = nn.Sequential(
            nn.Linear(in_features=enc_model_dim, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True))
    
            self.relu = self.activations[activation]
            self.sigmoid = self.activations['sigmoid']

        
        def forward(self, x, h, m, d, x_m, length):
                emb = []        
                for i, init_fc in enumerate(self.init_fc_list):
                    emb.append(init_fc(x[:, :, i]))
                emb = torch.stack(emb, 2)
                emb = self.init_ln(emb)
                emb = self.init_relu(emb).permute(0,2,1)
                output, _ = self.transformer_encoder(emb)
                output = self.classifier(output[:,0,:])
                return self.sigmoid(output)




