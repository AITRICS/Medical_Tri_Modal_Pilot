import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from builder.models.src.transformer.encoder import TrimodalTransformerEncoder_MT
from builder.models.src.transformer.module import LayerNorm
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from builder.models.src.vision_transformer import vit_b_16_m, ViT_B_16_Weights
from builder.models.src.swin_transformer import swin_t_m, Swin_T_Weights

class FEATURE_TEMPORAL_T_LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temporal_config = args.temporal
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
        
        ########## ########## Encoders ########## ##########
        ########## Spatio ##########
        # sharing VSLT encoder
        self.vslt_enc = nn.Sequential(
                                    nn.Linear(self.t_len, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.model_dim, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                    )
        
        self.instance_graph_transformer = TransformerEncoder(
                                        d_input=self.model_dim,
                                        n_layers=self.num_layers,
                                        n_head=self.num_heads,
                                        d_model=self.model_dim,
                                        d_ff=self.model_dim*4,
                                        dropout=self.dropout,
                                        pe_maxlen=25,
                                        use_pe=False,
                                        classification=True,
                                        mask=False)
        
        # TXT encoder
        datasetType = args.train_data_path.split("/")[-2]
        if datasetType == "mimic_icu": # BERT
            self.txt_embedding = nn.Embedding(30000, self.model_dim)
        elif datasetType == "sev_icu":
            raise NotImplementedError
        
        # Image encoder
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
        self.linear = nn.Linear(768,self.model_dim)     
        self.flatten = nn.Flatten(1,2)
        
        ########## Temporal ##########
        if self.temporal_config == "LSTM":
            self.temporal = nn.LSTM(input_size=self.model_dim, hidden_size=self.model_dim,
                            num_layers=2, batch_first=True)
            
        elif self.temporal_config == "BLSTM":
            self.temporal = nn.LSTM(input_size=self.model_dim, hidden_size=self.model_dim,
                            num_layers=2, batch_first=True, bidirectional=True)
            
        elif self.temporal_config == "transformer":
            self.temporal = TransformerEncoder(
                d_input=self.model_dim,
                                        n_layers=self.num_layers,
                                        n_head=self.num_heads,
                                        d_model=self.model_dim,
                                        d_ff=self.model_dim*4,
                                        dropout=self.dropout,
                                        pe_maxlen=5000,
                                        use_pe=True,
                                        classification=True,
                                        mask=True
            )
        elif self.temporal_config == "transformer_triangular":
            self.temporal = TransformerEncoder_Triangular(
                d_input=self.model_dim,
                                        n_layers=self.num_layers,
                                        n_head=self.num_heads,
                                        d_model=self.model_dim,
                                        d_ff=self.model_dim*4,
                                        dropout=self.dropout,
                                        pe_maxlen=5000,
                                        use_pe=True,
                                        classification=True,
                                        mask=True
            )

        ##### Classifier
        self.layer_norm_final = nn.LayerNorm(self.model_dim)
        self.fc_list = nn.Sequential(
            nn.Linear(in_features=self.model_dim, out_features= self.classifier_nodes, bias=True),
            # nn.BatchNorm1d(self.classifier_nodes),
            nn.LayerNorm(self.classifier_nodes),
            self.relu,
            nn.Linear(in_features=self.classifier_nodes, out_features= 1,  bias=True))
        
        
    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, missing, f_indices):
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        vslt_embedding = self.vslt_enc(x)

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
            
            
        if self.temporal_config == "LSTM":
            pass
        elif self.temporal_config == "BLSTM":
            pass
        elif self.temporal_config == "transformer" or self.temporal_config == "transformer_triangular":
            context_vector, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, img_embedding, txt_embedding], 
                fixed_lengths = [vslt_embedding.size(1), img_embedding.size(1), txt_embedding.size(1)],
                varying_lengths = [input_lengths, torch.tensor(img_embedding.size(1)).repeat(img_embedding.size(0)), txt_lengths+2],
                fusion_idx = None,
                missing=missing
            )
            
        context_vector = self.layer_norm_final(context_vector)
        output = self.fc_list(context_vector)
        
        return output, None