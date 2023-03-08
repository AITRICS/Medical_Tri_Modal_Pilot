import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *

class NEW_TXT_TRANS_VSLT_FTRANS_TOCONCAT(nn.Module):
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
            use_pe = False,
            classification = True,
            mask = False
        )

        self.txt_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.txt_num_layers,
            n_head = self.txt_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 2,
            dropout = self.txt_dropout,
            pe_maxlen = 600,
            use_pe = True,
            classification = True,
            mask = True
        )
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim * 2 + 2, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        self.relu = self.activations[activation]
        self.sigmoid = self.activations['sigmoid']


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
        
        self.init_fc_list = nn.ModuleList()
        for _ in range(self.num_nodes):
            self.init_fc_list.append(nn.Linear(self.t_len, self.model_dim))
        self.init_ln = nn.LayerNorm(self.num_nodes+2)
        self.init_relu = nn.ReLU(inplace=True)

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        vslt_embedding = []
        for i, init_fc in enumerate(self.init_fc_list):
            vslt_embedding.append(init_fc(x[:,:,i]))
        
        vslt_embedding.append(self.age_encoder(age.unsqueeze(1).unsqueeze(2)).squeeze(1))
        vslt_embedding.append(self.gender_encoder(gen.unsqueeze(1).unsqueeze(2)).squeeze(1))
        
        vslt_embedding = torch.stack(vslt_embedding, 2)
        vslt_embedding = self.init_ln(vslt_embedding)
        vslt_embedding = self.init_relu(vslt_embedding).permute(0,2,1)
        
        vslt_output, _ = self.vslt_encoder(vslt_embedding)
        vslt_cls_output = vslt_output[:, 0]

        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)

        txt_output, _ = self.txt_encoder(txt_embedding, input_lengths = txt_lengths + 2)
        txt_cls_output =  txt_output[:, 0]

        final_output = torch.cat([txt_cls_output, vslt_cls_output], 1)
        
        ageTensor = age.unsqueeze(1)
        genTensor = gen.unsqueeze(1)

        classInput = torch.cat((final_output, ageTensor, genTensor), 1)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)
        output = self.sigmoid(output)

        return output