import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from builder.models.src.transformer.mbt_encoder import BimodalTransformerEncoder_MBT
from builder.models.src.transformer.module import LayerNorm
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from builder.models.src.vision_transformer import vit_b_16_m, ViT_B_16_Weights
from builder.models.src.swin_transformer import swin_t_m, Swin_T_Weights
from builder.models.src.reports_transformer_decoder import TransformerDecoder
from transformers import AutoTokenizer

class BI_VSLTTXT_MBT_V1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        ##### Configuration
        self.output_dim = args.output_dim
        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout
        self.idx_order = torch.arange(0, args.batch_size).type(torch.LongTensor)
        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size

        self.device = args.device
        self.vslt_input_size = len(args.vitalsign_labtest)
        self.n_modality = len(args.input_types.split("_"))
        self.bottlenecks_n = 4
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
        self.fusion_transformer = BimodalTransformerEncoder_MBT(
            batch_size = args.batch_size,
            n_modality = 2,
            bottlenecks_n = 4,      # https://arxiv.org/pdf/2107.00135.pdf # according to section 4.2 implementation details
            fusion_startidx = args.mbt_fusion_startIdx,
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            txt_idx = 1,
            pe_maxlen = 2500,
            use_pe = [vslt_pe, True],
            mask = [True, True],
        )

        ##### Classifier
        if self.args.vslt_type == "QIE":
            classifier_dim = self.model_dim
        else:
            classifier_dim = self.model_dim*2
        self.layer_norms_after_concat = nn.LayerNorm(self.model_dim)
        self.fc_list = nn.Sequential(
        nn.Linear(in_features=classifier_dim, out_features= self.model_dim, bias=True),
        nn.BatchNorm1d(self.model_dim),
        self.activations[activation],
        nn.Linear(in_features=self.model_dim, out_features= 1,  bias=True))

        self.relu = self.activations[activation]
        self.img_feat = torch.Tensor([18]).repeat(self.args.batch_size).unsqueeze(1).type(torch.LongTensor).to(self.device, non_blocking=True)
        self.txt_feat = torch.Tensor([19]).repeat(self.args.batch_size).unsqueeze(1).type(torch.LongTensor).to(self.device, non_blocking=True)
        
        if "rmse" in self.args.auxiliary_loss_type:
            self.rmse_layer = nn.Linear(in_features=classifier_dim, out_features= 1, bias=True)
        
    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, missing, f_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths):
        # x-TIE:  torch.Size([bs, vslt_len, 3])
        # x-Carryforward:  torch.Size([bs, 24, 16])
        # txts:  torch.Size([bs, 128, 768])         --> ([bs, 128, 256])
        # img:  torch.Size([bs, 1, 224, 224])       --> ([bs, 49, 256])
        # age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        # gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)    
        # x = torch.cat([x, age, gen], axis=2)
        
        demographic = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1)
        demo_embedding = self.ie_demo(demographic)
                                
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
            vslt_embedding = value_embedding + time_embedding + feat_embedding +demo_embedding
        if self.args.berttype == "bert":
            txts = txts.type(torch.LongTensor).cuda()
        txt_embedding = self.txt_embedding(txts)
            
        if self.args.imgtxt_time == 1:
            if self.args.vslt_type == "QIE":
                demographic_it = torch.cat([age.unsqueeze(1), gen.unsqueeze(1)], dim=1).unsqueeze(1)
                demo_embedding_it = self.ie_demo(demographic_it)
                txt_embedding = txt_embedding + self.ie_time(txt_time.unsqueeze(1)).unsqueeze(1) + self.ie_feat(self.txt_feat) + demo_embedding_it               
            else:
                txt_embedding = txt_embedding + self.ie_time(txt_time.unsqueeze(1)).unsqueeze(1) + self.ie_feat(self.txt_feat)
        
        outputs, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, txt_embedding], 
                                      fixed_lengths = [vslt_embedding.size(1), txt_embedding.size(1)],
                                      varying_lengths = [input_lengths, txt_lengths+2],
                                      fusion_idx = None,
                                      missing=missing
                                      )
        outputs_stack = torch.stack([outputs[0][:, 0, :], outputs[1][:, 0, :]]) 
        bi_mean = torch.mean(outputs_stack, dim=0) 
        all_cls_stack = torch.stack([bi_mean, outputs[0][:, 0, :]])
        output = all_cls_stack[missing, self.idx_order]
        
        classInput = self.layer_norms_after_concat(output)
        if self.args.vslt_type != "QIE":
            classInput = torch.cat([classInput, demo_embedding], dim=1)
        output = self.fc_list(classInput).squeeze()
        
        if "rmse" in self.args.auxiliary_loss_type:
            output2 = self.rmse_layer(classInput).squeeze()
        else:
            output2 = None
        
        output3 = None
        return output, output2, output3