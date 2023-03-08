import torch.nn as nn
import torch
from builder.models.src.transformer import *
import pickle
import matplotlib.pyplot as plt
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from bpe import Encoder

class T_TRANSFORMER_V1(nn.Module):
    def __init__(self, args):
        super(T_TRANSFORMER_V1, self).__init__()      
        self.args = args

        self.num_layers = args.txt_num_layers
        self.dropout = args.txt_dropout
        self.model_d = args.txt_model_dim
        self.num_heads = args.txt_num_heads
        self.classifier_nodes = args.txt_classifier_nodes

        self.transformer_encoder = TransformerEncoder(
            d_input = self.model_d,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_d,
            d_ff = self.model_d * 2,
            dropout = self.dropout,
            pe_maxlen = 600,
            use_pe = True,
            classification = True
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.model_d, out_features=self.classifier_nodes, bias=True),
            nn.BatchNorm1d(self.classifier_nodes),
            nn.ReLU(),
            nn.Linear(in_features=self.classifier_nodes, out_features=1, bias=True)
        )

        datasetType = args.train_data_path.split("/")[-2]
        if args.txt_tokenization == "word":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(1620, self.model_d)
            elif datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(3720, self.model_d)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(45282, self.mode_d)
        elif args.txt_tokenization == "character":
            if datasetType == "mimic_icu" or datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(42, self.model_d)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(1130, self.model_d)
        elif args.txt_tokenization == "bpe":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(8005, self.model_d)
            elif datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(8005, self.model_d)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(8005, self.model_d)
        elif args.txt_tokenization == "bert":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(30000, self.model_d)
            elif datasetType == "sev_icu":
                raise NotImplementedError
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_d)).to(self.args.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, input_lengths):
        txt_embedding = self.linear_embedding(x)
        txt_output, _ = self.transformer_encoder(txt_embedding, input_lengths+1)
        txtClsOutput = txt_output[:,0]
        output = self.classifier(txtClsOutput)
        sigOut = self.sigmoid(output)
        return sigOut