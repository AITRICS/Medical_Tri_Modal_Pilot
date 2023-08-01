import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from builder.models.src.transformer.encoder import TrimodalTransformerEncoder_MT
from builder.models.src.transformer.module import LayerNorm
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from builder.models.src.vision_transformer import vit_b_16_m, ViT_B_16_Weights
from builder.models.src.swin_transformer import swin_t_m, Swin_T_Weights
from builder.models.src.reports_transformer_decoder import TransformerDecoder
from transformers import AutoTokenizer

# early fusion

class BITXT_MT_V1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        ##### Configuration
        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout
        if args.output_type =="intubation":
            self.output_dim = 1#args.output_dim
        else:
            self.output_dim = 2#args.output_dim

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
            self.vslt_enc = nn.Sequential(
                                        nn.Linear(self.num_nodes, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
            vslt_pe = True
            
        elif args.vslt_type == "TIE" or args.vslt_type == "QIE":
            vslt_pe = False
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
            
        if args.berttype == "bert": # BERT
            self.txt_embedding = nn.Embedding(30000, self.model_dim)
        elif args.berttype == "biobert": # BIOBERT
            self.txt_embedding = nn.Linear(768, self.model_dim)
        
        
        ##### Fusion Part
        self.fusion_transformer = TrimodalTransformerEncoder_MT(
            batch_size = args.batch_size,
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            fusion_startidx = args.mbt_fusion_startIdx,
            d_ff = self.model_dim * 4,
            n_modality = self.n_modality,
            dropout = self.dropout,
            pe_maxlen = 2500,
            use_pe = [vslt_pe, True],
            mask = [True, True],
            txt_idx = 1,
        )

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
        
        # if "rmse" in self.args.auxiliary_loss_type:
        #     self.rmse_layer = nn.Linear(in_features=classifier_dim, out_features= 1, bias=True)
        
        self.fixed_lengths = [0, 25]
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

        txt_embedding = self.txt_embedding(txts)
        
        txt_embedding = txt_embedding + self.ie_time(txt_time.unsqueeze(1)).unsqueeze(1) + self.ie_feat(self.txt_feat)
        
        context_vector, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, txt_embedding], 
            fixed_lengths = [vslt_embedding.size(1), txt_embedding.size(1)],
            varying_lengths = [input_lengths, txt_lengths+2],
            fusion_idx = None,
            missing=missing
        )
        final_cls_output = context_vector[:,0,:]
            
        classInput = self.layer_norm_final(final_cls_output)
        if self.args.vslt_type != "QIE":
            classInput = torch.cat([classInput, demo_embedding], dim=1)
        output = self.fc_list(classInput)
        output2 = None
        output3 = None
 
        return output[:,0], output2, output3