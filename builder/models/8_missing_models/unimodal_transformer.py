import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *

class UNIMODAL_TRANSFORMER(nn.Module):
        def __init__(self, args):
            super().__init__()      
            self.args = args
            ##### Configuration
            self.num_layers = args.transformer_num_layers
            self.num_heads = args.transformer_num_head
            self.model_dim = args.transformer_dim
            self.dropout = args.dropout
            self.output_dim = args.output_dim

            self.num_nodes = len(args.vitalsign_labtest)
            self.t_len = args.window_size

            self.device = args.device
            self.vslt_input_size = len(args.vitalsign_labtest)
            self.n_modality = len(args.input_types.split("_"))
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
            self.relu = self.activations[activation]
            
            ##### Encoders
            if args.vslt_type == "carryforward":
                usepe = True
                self.vslt_enc = nn.Sequential(
                                            nn.Linear(self.num_nodes, self.model_dim),
                                            nn.LayerNorm(self.model_dim),
                                            nn.ReLU(inplace=True),
                        )
                vslt_pe = True
                
            elif args.vslt_type == "TIE" or args.vslt_type == "QIE":
                usepe = False
                self.ie_vslt = nn.Sequential(
                                            nn.Linear(1, self.model_dim),
                                            nn.LayerNorm(self.model_dim),
                                            nn.ReLU(inplace=True),
                        )
                self.ie_time = nn.Sequential(
                                            nn.Linear(1, self.model_dim),
                                            nn.LayerNorm(self.model_dim),
                                            nn.ReLU(inplace=True),
                        )
                self.ie_feat = nn.Embedding(20, self.model_dim)
            self.ie_demo = nn.Sequential(
                                        nn.Linear(2, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
            self.transformer_encoder = TransformerEncoder(
                                        d_input=self.model_dim,
                                        n_layers=self.num_layers,
                                        n_head=self.num_heads,
                                        d_model=self.model_dim,
                                        d_ff=self.model_dim*4,
                                        dropout=self.dropout,
                                        pe_maxlen=2000,
                                        use_pe=usepe,
                                        classification=True)
            
            ##### Classifier
            if self.args.vslt_type == "QIE":
                classifier_dim = self.model_dim
            else:
                classifier_dim = self.model_dim*2
            self.layer_norm_final = nn.LayerNorm(self.model_dim)
            self.fc_list = nn.Sequential(
            nn.Linear(in_features=classifier_dim, out_features= self.model_dim, bias=True),
            nn.BatchNorm1d(self.model_dim),
            self.activations[activation],
            nn.Linear(in_features=self.model_dim, out_features= self.output_dim,  bias=True))
            
            self.fixed_lengths = [0, 25]
            self.img_feat = torch.Tensor([18]).repeat(self.args.batch_size).unsqueeze(1).type(torch.LongTensor).to(self.device, non_blocking=True)
            self.txt_feat = torch.Tensor([19]).repeat(self.args.batch_size).unsqueeze(1).type(torch.LongTensor).to(self.device, non_blocking=True)
        
        
        def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, missing, f_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths):
            if self.args.vslt_type == "carryforward":
                demographic = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1)
                vslt_embedding = self.vslt_enc(x)
                demo_embedding = self.ie_demo(demographic)
            elif self.args.vslt_type == "TIE": # [seqlen x 3] 0: time, 1: value, 2: feature    
                demographic = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1)
                value_embedding = self.ie_vslt(x[:,:,1].unsqueeze(2))
                time_embedding = self.ie_time(x[:,:,0].unsqueeze(2))
                feat = x[:,:,2].type(torch.IntTensor).to(self.device)
                feat_embedding = self.ie_feat(feat)
                vslt_embedding = value_embedding + time_embedding + feat_embedding
                demo_embedding = self.ie_demo(demographic)
            elif self.args.vslt_type == "QIE":
                value_embedding = self.ie_vslt(x[:,:,1].unsqueeze(2))
                time_embedding = self.ie_time(x[:,:,0].unsqueeze(2))
                feat = x[:,:,2].type(torch.IntTensor).to(self.device)
                feat_embedding = self.ie_feat(feat)
                demographic = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1).unsqueeze(1).repeat(1,x.size(1),1)
                demo_embedding = self.ie_demo(demographic)
                vslt_embedding = value_embedding + time_embedding + feat_embedding + demo_embedding
            context_vector, _ = self.transformer_encoder(vslt_embedding, input_lengths=input_lengths+1)

            final_cls_output = context_vector[:,0,:]
            
            classInput = self.layer_norm_final(final_cls_output)
            if self.args.vslt_type != "QIE":
                classInput = torch.cat([classInput, demo_embedding], dim=1)
            output = self.fc_list(classInput) 
            return output, None



