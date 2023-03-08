import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *

# early fusion

class MT_V1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        ##### Configuration
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
        self.vslt_enc = nn.Sequential(
                            nn.Linear(self.num_nodes+2, self.model_dim),
                            nn.LayerNorm(self.model_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.model_dim, self.model_dim),
                            nn.LayerNorm(self.model_dim),
                            nn.ReLU(inplace=True),
        )
    
        datasetType = args.train_data_path.split("/")[-2]
        if datasetType == "mimic_icu": # BERT
            self.txt_embedding = nn.Embedding(30000, self.model_dim)
        elif datasetType == "sev_icu":
            raise NotImplementedError

        ##### Fusion Part
        self.fusion_transformer = MultimodalTransformerEncoder(
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            n_modality = self.n_modality,
            dropout = self.dropout,
            pe_maxlen = 2000,
            use_pe = True,
            classification = True,
            txt_idx = 1
        )

        ##### Classifier
        self.layer_norm_final = nn.LayerNorm(self.model_dim)
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim, out_features= self.classifier_nodes, bias=True),
            nn.BatchNorm1d(self.classifier_nodes),
            self.relu,
            nn.Linear(in_features=self.classifier_nodes, out_features= 1,  bias=True)))
        
        self.fixed_lengths = [0, 25]
        
    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, missing, f_indices):
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        vslt_embedding = self.vslt_enc(x)

        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)

        context_vector, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, txt_embedding], 
            fixed_lengths = [vslt_embedding.size(1), txt_embedding.size(1)],
            varying_lengths = [input_lengths, txt_lengths+2],
            fusion_idx = None
        ) # torch.Size([64, 155, 256])
        
        context_vector = self.layer_norm_final(context_vector[:,0,:])
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(context_vector))
        output = torch.stack(multitask_vectors) # torch.Size([12, 64, 1])
        
        return output, None