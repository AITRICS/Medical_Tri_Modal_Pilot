import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *

class BI_CROSS_TRANSFORMER_TXT_VSLT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model_dim = args.transformer_dim
        self.txt_num_layers = args.transformer_num_layers
        self.txt_num_heads = args.transformer_num_head
        self.txt_dropout = 0.1

        self.self_num_layers = args.transformer_num_layers
        self.self_num_heads = args.transformer_num_head
        self.self_model_dim = args.transformer_dim
        self.self_dropout = 0.1
        
        self.cross_num_layers = args.cross_transformer_num_layers
        self.cross_num_heads = args.cross_transformer_num_head
        self.cross_model_dim = args.cross_transformer_dim
        self.cross_dropout = 0.1

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

        self.vslt_input_size = len(args.vitalsign_labtest)        
        
        # self.init_fc = nn.Linear(self.num_nodes + 2, self.cross_model_dim)
        self.init_fc = nn.Sequential(
                                    nn.Linear(self.num_nodes+2, self.cross_model_dim),
                                    nn.LayerNorm(self.cross_model_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.cross_model_dim, self.cross_model_dim),
                                    nn.LayerNorm(self.cross_model_dim),
                                    nn.ReLU(inplace=True),
                )

        self.txt_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.txt_num_layers,
            n_head = self.txt_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.txt_dropout,
            pe_maxlen = 500,
            use_pe = True,
            classification = False
        )

        self.self_attn_encoder = TransformerEncoder(
            d_input = self.self_model_dim,
            n_layers = self.self_num_layers,
            n_head = self.self_num_heads,
            d_model = self.self_model_dim,
            d_ff = self.self_model_dim * 4,
            dropout = self.self_dropout,
            pe_maxlen = 500,
            use_pe = True,
            classification = False
        )
        
        self.cross_attn_encoder_vsltanchor = CrossTransformerEncoder(
            d_input = self.cross_model_dim,
            n_layers = self.cross_num_layers,
            n_head = self.cross_num_heads,
            d_model = self.cross_model_dim,
            d_ff = self.cross_model_dim * 4,
            dropout = self.cross_dropout,
            pe_maxlen = 500,
            use_pe = False,
            classification = False
        )
        
        self.cross_attn_encoder_txtanchor = CrossTransformerEncoder(
            d_input = self.cross_model_dim,
            n_layers = self.cross_num_layers,
            n_head = self.cross_num_heads,
            d_model = self.cross_model_dim,
            d_ff = self.cross_model_dim * 2,
            dropout = self.cross_dropout,
            pe_maxlen = 500,
            use_pe = False,
            classification = False,
            mask=False
        )
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim*2, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        self.relu = self.activations[activation]
        # self.sigmoid = self.activations['sigmoid']
        self.cls_tokens_vslt = nn.Parameter(torch.zeros(1, 1, self.self_model_dim))
        self.cls_tokens_txt = nn.Parameter(torch.zeros(1, 1, self.self_model_dim))

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        
        ### TXT transformer ###
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        cls_tokens_txt = self.cls_tokens_txt.repeat(txt_embedding.size(0), 1, 1)
        
        if self.self_num_layers > 0:
            txt_embedding = torch.cat([cls_tokens_txt, txt_embedding], axis=1)
            txt_output, _ = self.txt_encoder(txt_embedding, input_lengths = txt_lengths)
        else:
            txt_output = torch.cat([cls_tokens_txt, txt_embedding], axis=1)

        ### VSLT encoder ###
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        emb_vslt = self.init_fc(x)
        cls_tokens_vslt = self.cls_tokens_vslt.repeat(emb_vslt.size(0), 1, 1)
        if self.self_num_layers > 0:
            emb_vslt = torch.cat([cls_tokens_vslt, emb_vslt], axis=1)
            self_enc_output, _ = self.self_attn_encoder(emb_vslt,
                                            input_lengths = input_lengths)
        else:
            self_enc_output = torch.cat([cls_tokens_vslt, emb_vslt], axis=1)
            
        # print("#####cross#####")
        cross_enc_output_vslt, _ = self.cross_attn_encoder_vsltanchor(padded_q_input = self_enc_output, 
                                            padded_kv_input = txt_output, 
                                            q_input_lengths = self_enc_output.size(1),
                                            kv_input_lengths = txt_lengths+1)
        # print("#####cross2#####")
        cross_enc_output_txt, _ = self.cross_attn_encoder_txtanchor(padded_q_input = txt_output, 
                                            padded_kv_input = self_enc_output, 
                                            q_input_lengths = txt_output.size(1),
                                            kv_input_lengths = self_enc_output)
        finalClsOutput_vslt = cross_enc_output_vslt[:, 0]
        finalClsOutput_txt = cross_enc_output_txt[:, 0]
        finalClsOutput = torch.cat([finalClsOutput_vslt, finalClsOutput_txt], axis=1)
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(finalClsOutput))
        output = torch.stack(multitask_vectors)
        # output = self.sigmoid(output)
        return output