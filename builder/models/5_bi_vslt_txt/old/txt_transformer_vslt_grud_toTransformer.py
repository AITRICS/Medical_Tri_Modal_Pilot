import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.encoder import TransformerEncoder
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from control.config import args

class TXT_TRANSFORMER_VSLT_GRUD_TOTRANSFORMER(nn.Module):
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
        self.input_decay = nn.ModuleList()
        for _ in range(self.vslt_input_size):
            self.input_decay.append(nn.Linear(1,1))
        
        self.hidden_decay = nn.Linear(self.vslt_input_size, self.model_dim)
        self.gru = nn.GRUCell(self.vslt_input_size * 2, self.model_dim, True)

        self.pos_encoding = PositionalEncoding(d_model=self.model_dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_dim)).to(self.args.device)

        self.txt_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.txt_num_layers,
            n_head = self.txt_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 2,
            dropout = self.txt_dropout,
            pe_maxlen = 500,
            use_pe = True,
            classification = False
        )

        self.final_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 2,
            dropout = self.dropout,
            pe_maxlen = 500,
            use_pe = True,
            classification = True
        )
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim + 2, out_features= 64, bias=True),
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

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        x_d = []
        for i in range(x.shape[-1]):
                x_d.append(self.input_decay[i](d[:, :, i].unsqueeze(-1)))
        x_d = torch.cat(x_d, dim=-1)
        x_d = torch.exp(-self.relu(x_d))
        x_m = x_m.view(1, 1, -1)

        x = m * x + (1 - m) * x_d * x + (1 - m) * (1 - x_d) * x_m

        grud_output = []
 
        for i in range(x.size(1)):
                h_d = self.hidden_decay(d[:, i])
                h_d = torch.exp(-self.relu(h_d))
                h = h_d * h
                x_t = torch.cat((x[:, i], m[:, i]), dim=-1)
                h = self.gru(x_t, h)
                grud_output.append(h)
        emb = torch.stack(grud_output, dim=1)
        
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        txt_output, _ = self.txt_encoder(txt_embedding, input_lengths = txt_lengths)

        finalInputs = torch.cat((emb, txt_output), dim=1)
        finalOutput, _ = self.final_encoder(finalInputs, input_lengths = txt_lengths + emb.shape[1] + 1)
        finalClsOutput = finalOutput[:, 0]
        
        ageTensor = age.unsqueeze(1)
        genTensor = gen.unsqueeze(1)

        classInput = torch.cat((finalClsOutput, ageTensor, genTensor), 1)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)

        output = self.sigmoid(output)

        return output