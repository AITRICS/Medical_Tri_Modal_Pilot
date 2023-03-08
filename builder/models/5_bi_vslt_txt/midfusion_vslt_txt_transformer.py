import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *

# mid concat V-trans T-trans

class MIDFUSION_VSLT_TXT_TRANSFORMER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.txt_num_layers = args.txt_num_layers
        self.txt_num_heads = args.txt_num_heads
        self.txt_dropout = args.txt_dropout

        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout

        self.final_num_layers = args.cross_transformer_num_layers
        self.final_num_heads = args.cross_transformer_num_head
        self.final_dropout = args.final_dropout

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

        self.vslt_input_size = len(args.vitalsign_labtest)

        self.age_encoder = nn.Linear(1, self.model_dim)
        self.gender_encoder = nn.Linear(1, self.model_dim)
        self.pos_encoding = PositionalEncoding(d_model=self.model_dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_dim)).to(self.args.device)

        self.vslt_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 600,
            use_pe = True,
            classification = False,
            mask = True
        )

        self.txt_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.txt_num_layers,
            n_head = self.txt_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.txt_dropout,
            pe_maxlen = 600,
            use_pe = True,
            classification = False,
            mask = True
        )

        self.final_encoder = BimodalTransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.final_num_layers,
            n_head = self.final_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.final_dropout,
            pe_maxlen = 600,
            use_pe = False,
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
        # self.sigmoid = self.activations['sigmoid']


        datasetType = args.train_data_path.split("/")[-2]
        if args.txt_tokenization == "word":
            if datasetType == "mimic_icu":
                self.txt_embedding = nn.Embedding(1620, self.model_dim)
            elif datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(3720, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(45282, self.model_dim)
        elif args.txt_tokenization == "character":
            if datasetType == "mimic_icu" or datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(42, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(1130, self.model_dim)
        elif args.txt_tokenization == "bpe":
            if datasetType == "mimic_icu":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
            elif datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
        elif args.txt_tokenization == "bert":
            if datasetType == "mimic_icu":
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

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        vslt_embedding = self.init_fc(x)
        vslt_output, _ = self.vslt_encoder(vslt_embedding, input_lengths = input_lengths - 1)

        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        txt_output, _ = self.txt_encoder(txt_embedding, input_lengths = txt_lengths + 1)

        final_input = torch.cat([vslt_output, txt_output], 1)
        final_output, _ = self.final_encoder(final_input, 
            first_size = vslt_output.size(1),
            first_lengths = input_lengths + 1,
            second_lengths = txt_lengths + 2
        )
        final_cls_output = final_output[:, 0, :]

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(final_cls_output))
        output = torch.stack(multitask_vectors)
        # output = self.sigmoid(output)

        return output