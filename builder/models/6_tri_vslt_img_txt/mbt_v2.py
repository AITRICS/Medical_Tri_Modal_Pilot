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

# late concat V-trans T-trans

class MBT_V2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size = 224
        img_hidden_size = 256
        patch_size = 16
        img_num_heads = 4
        pos_embed = "conv"

        self.n_modality = 3

        self.num_layers = 12
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout

        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size

        self.device = args.device
        self.classifier_nodes = 64

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
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=img_hidden_size,
            num_heads=img_num_heads,
            pos_embed=pos_embed,
            dropout_rate=0,
            spatial_dims=2,
            )                
        
        self.vslt_encoder = TrimodalTransformerEncoder_MBT(
            n_modality = self.n_modality,
            bottlenecks_n = 4,      # https://arxiv.org/pdf/2107.00135.pdf # according to section 4.2 implementation details
            fusion_startidx = args.mbt_fusion_startIdx,
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 600,
            use_pe = [True, True],
            mask = [True, True],
        )
        
        self.layer_norms_after_concat = nn.LayerNorm(self.model_dim * 3)
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim * 3, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        self.relu = self.activations[activation]
        # self.sigmoid = self.activations['sigmoid']

        datasetType = args.train_data_path.split("/")[-2]
        if args.txt_tokenization == "word":
            if "mimic_icu" in datasetType:
                self.txt_embedding = nn.Embedding(1620, self.model_dim)
            elif datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(3720, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(45282, self.model_dim)
        elif args.txt_tokenization == "character":
            if "mimic_icu" in datasetType or datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(42, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(1130, self.model_dim)
        elif args.txt_tokenization == "bpe":
            if "mimic_icu" in datasetType:
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
            elif datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
        elif args.txt_tokenization == "bert":
            if "mimic_icu" in datasetType:
                self.txt_embedding = nn.Embedding(30000, self.model_dim)
            elif datasetType == "sev_icu":
                raise NotImplementedError
        
        # self.linear_embedding = nn.Linear(self.num_nodes, self.model_dim)
        self.init_fc = nn.Sequential(
                                    nn.Linear(self.num_nodes+2, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.model_dim, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                )


    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, img, txts, txt_lengths):
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        vslt_embedding = self.init_fc(x)
        txts = txts.type(torch.IntTensor).to(self.device)
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
        
        outputs, _ = self.vslt_encoder([vslt_embedding, img_embedding, txt_embedding], 
                                      lengths = [input_lengths, torch.tensor([img_embedding.size(1)]*img_embedding.size(0)) + 2, txt_lengths + 2])
        final_output = torch.cat([outputs[i][:, 0, :] for i in range(self.n_modality)], dim=1)
        
        classInput = self.layer_norms_after_concat(final_output)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)
        
        exit(1)

        return output