import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

class EARLYFUSION_IMG_VSLT_V2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        img_size = 224
        img_hidden_size = 256
        patch_size = 16
        img_num_heads = 4
        pos_embed = "conv"

        self.num_layers = args.transformer_num_layers
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

        #Image
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
        
        self.vslt_input_size = len(args.vitalsign_labtest)
        ####4개 필요 없음 뒤에 안쓰임
        #self.age_encoder = nn.Linear(1, self.model_dim)
        #self.gender_encoder = nn.Linear(1, self.model_dim)
        #self.pos_encoding = PositionalEncoding(d_model=self.model_dim)
        #self.layer_norm_in = nn.LayerNorm(self.model_dim)
        #self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_dim)).to(self.args.device)

        self.final_encoder = BimodalTransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 600,
            use_pe = True,
            classification = True,
            mask = True
        )

        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        self.relu = self.activations[activation]
        self.sigmoid = self.activations['sigmoid']

        #self.linear_embedding = nn.Linear(self.num_nodes, self.model_dim)
        self.init_fc = nn.Sequential(
                                    nn.Linear(self.num_nodes+2, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.model_dim, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                )

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, img):
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        vslt_embedding = self.init_fc(x)

        # txts = txts.type(torch.IntTensor).to(self.device)
        img_embedding = self.patch_embedding(img)

        final_input = torch.cat([vslt_embedding, img_embedding], 1)
        final_output, _ = self.final_encoder(final_input, 
            first_size = vslt_embedding.size(1),
            first_lengths = input_lengths + 1,
            second_lengths = torch.tensor([img_embedding.size(1)]*img_embedding.size(0)) + 2
        )
        
        final_output = final_output[:, 0]

        #ageTensor = age.unsqueeze(1)
        #genTensor = gen.unsqueeze(1)

        #classInput = torch.cat((final_output, ageTensor, genTensor), 1)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(final_output))
        output = torch.stack(multitask_vectors)
        #output = self.sigmoid(output)
        # exit(1)
        return output