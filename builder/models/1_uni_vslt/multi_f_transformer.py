import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from builder.models.src.transformer import *

class MULTI_F_TRANSFORMER(nn.Module):
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

                self.init_fc_list = nn.ModuleList()
                for _ in range(self.num_nodes):
                        if args.enc_depth == 1:
                                self.init_fc_list.append(nn.Linear(self.t_len, enc_model_dim))
                        elif args.enc_depth == 2:
                                self.init_fc_list.append(nn.Sequential(
                                                nn.Linear(self.t_len, enc_model_dim),
                                                nn.LayerNorm(enc_model_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(enc_model_dim, enc_model_dim),
                                ))
                        elif args.enc_depth == 3:
                                self.init_fc_list.append(nn.Sequential(
                                                nn.Linear(self.t_len, enc_model_dim),
                                                nn.LayerNorm(enc_model_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(enc_model_dim, enc_model_dim),
                                                nn.LayerNorm(enc_model_dim),
                                                nn.ReLU(inplace=True),
                                                # nn.Linear(enc_model_dim, enc_model_dim),
                                ))
                        else:            
                                raise ValueError('invaliud enc depth option!')
                self.age_encoder = nn.Linear(1, enc_model_dim)
                self.gender_encoder = nn.Linear(1, enc_model_dim)

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
            
                self.fc_list = nn.ModuleList()
                for _ in range(12):
                        self.fc_list.append(nn.Sequential(
                        nn.Linear(in_features=enc_model_dim, out_features= 64, bias=True),
                        nn.BatchNorm1d(64),
                        self.activations[activation],
                        nn.Linear(in_features=64, out_features= 1,  bias=True)))

                self.relu = self.activations[activation]
                # self.sigmoid = self.activations['sigmoid']

        def forward(self, x, h, m, d, x_m, age, gen, input_lengths):
                emb = []        
                for i, init_fc in enumerate(self.init_fc_list):
                        emb.append(init_fc(x[:, :, i]))
                emb.append(self.age_encoder(age.unsqueeze(1).unsqueeze(2)).squeeze(1))
                emb.append(self.gender_encoder(gen.unsqueeze(1).unsqueeze(2)).squeeze(1))

                emb = torch.stack(emb, 2)
                
                output, _ = self.transformer_encoder(emb.permute(0,2,1))
                multitask_vectors = []
                for i, fc in enumerate(self.fc_list):
                        multitask_vectors.append(fc(output[:,0,:]))
                output = torch.stack(multitask_vectors)    

                # return self.sigmoid(output)
                return output




