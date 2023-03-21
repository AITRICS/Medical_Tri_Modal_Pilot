import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from builder.models.src.transformer.mbt_encoder import TrimodalTransformerEncoder_MBT
from builder.models.src.transformer.module import LayerNorm
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from builder.models.src.vision_transformer import vit_b_16_m, ViT_B_16_Weights
from builder.models.src.swin_transformer import swin_t_m, Swin_T_Weights

class TRI_MBT_V1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        ##### Configuration
        self.img_size = args.image_size
        self.patch_size = 16
        self.img_num_heads = 4
        self.pos_embed = "conv"
        
        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout

        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size

        self.device = args.device
        self.classifier_nodes = 64
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
                                        nn.Linear(self.num_nodes+2, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
        elif args.vslt_type == "TIE":
            self.tie_vslt = nn.Sequential(
                                        nn.Linear(1, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
            self.tie_time = nn.Sequential(
                                        nn.Linear(1, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
            self.tie_feat = nn.Sequential(
                                        nn.Linear(1, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
            self.tie_demo = nn.Sequential(
                                        nn.Linear(2, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                    )
        
        if args.berttype == "bert": # BERT
            self.txt_embedding = nn.Embedding(30000, self.model_dim)
        elif args.berttype == "biobert": # BIOBERT
            self.txt_embedding = nn.Linear(768, self.model_dim)
        
        self.img_model_type = args.img_model_type
        self.img_pretrain = args.img_pretrain
        if self.img_model_type == "vit":
            if self.img_pretrain == "Yes":
                self.img_encoder = vit_b_16_m(weights = ViT_B_16_Weights.IMAGENET1K_V1)#vit_b_16
            else:
                self.img_encoder = vit_b_16_m(weights = None)
        elif self.img_model_type == "swin":
            if self.img_pretrain =="Yes":
                self.img_encoder = swin_t_m(weights = Swin_T_Weights.IMAGENET1K_V1)#Swin_T_Weights.IMAGENET1K_V1
            else:
                self.img_encoder = swin_t_m(weights = None)
        else:
            self.patch_embedding = PatchEmbeddingBlock(
            in_channels=1,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=self.model_dim,
            num_heads=self.img_num_heads,
            pos_embed=self.pos_embed,
            dropout_rate=0,
            spatial_dims=2,
            )           
        self.linear = nn.Linear(768,256)     
        self.flatten = nn.Flatten(1,2)

        ##### Fusion Part
        self.fusion_transformer = TrimodalTransformerEncoder_MBT(
            batch_size = args.batch_size,
            n_modality = self.n_modality,
            bottlenecks_n = self.bottlenecks_n,      # https://arxiv.org/pdf/2107.00135.pdf # according to section 4.2 implementation details
            fusion_startidx = args.mbt_fusion_startIdx,
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 10000,
            use_pe = [True, True, True],
            mask = [True, False, True],
        )

        ##### Classifier
        self.layer_norms_after_concat = nn.LayerNorm(self.model_dim)
        self.fc_list = nn.Sequential(
        nn.Linear(in_features=self.model_dim, out_features= self.model_dim, bias=True),
        nn.BatchNorm1d(self.model_dim),
        self.activations[activation],
        nn.Linear(in_features=self.model_dim, out_features= 1,  bias=True))

        self.relu = self.activations[activation]

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, missing, f_indices):
        # x-TIE:  torch.Size([bs, vslt_len, 3])
        # x-Carryforward:  torch.Size([bs, 24, 16])
        # txts:  torch.Size([bs, 128, 768])         --> ([bs, 128, 256])
        # img:  torch.Size([bs, 1, 224, 224])       --> ([bs, 49, 256])
        
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)    
        if self.args.vslt_type == "carryforward":
            x = torch.cat([x, age, gen], axis=2)
            vslt_embedding = self.vslt_enc(x)
        else:
            value_embedding = self.tie_vslt(x[:,:,1].unsqueeze(2))
            time_embedding = self.tie_time(x[:,:,0].unsqueeze(2))
            feat_embedding = self.tie_feat(x[:,:,2].unsqueeze(2))
            demographic = torch.cat([age, gen], axis=2)
            demo_embedding = self.tie_demo(demographic)
            vslt_embedding = value_embedding + time_embedding + feat_embedding + demo_embedding

        txt_embedding = self.txt_embedding(txts)
        
        if self.img_model_type == "vit":
            img_embedding = self.img_encoder(img)#[16, 1000] #ViT_B_16_Weights.IMAGENET1K_V1
            img_embedding = self.linear(img_embedding)
        elif self.img_model_type == "swin":
            img_embedding = self.img_encoder(img)
            img_embedding = self.flatten(img_embedding)
            img_embedding = self.linear(img_embedding)     
        else:
            img_embedding = self.patch_embedding(img)
            
        outputs, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, img_embedding, txt_embedding], 
                                      fixed_lengths = [vslt_embedding.size(1), img_embedding.size(1), txt_embedding.size(1)],
                                      varying_lengths = [input_lengths, img_embedding.size(1), txt_lengths+2],
                                      fusion_idx = None,
                                      missing=missing
                                      )
        
        outputs_list = torch.stack([outputs[0][:, 0, :], outputs[1][:, 0, :], outputs[2][:, 0, :]]) # vslt, img, txt
        classInput = self.layer_norms_after_concat(outputs_list).reshape(-1, self.model_dim)
        output_vectors = self.fc_list(classInput)
        output = torch.mean(output_vectors.reshape(3, -1), dim=0)

        return output, None