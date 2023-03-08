import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import importlib
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from monai.networks.nets import ViT

swin_transformer_module = importlib.import_module("builder.models.2_uni_image.swin_transformer")
SWIN_TRANSFORMER_MODEL = getattr(swin_transformer_module, "SWIN_TRANSFORMER_MODEL")
resnet_module = importlib.import_module("builder.models.2_uni_image.resnet_enc")
RESNET = getattr(resnet_module, "ResNet")
RESNET_BLOCK = getattr(resnet_module, "ResNetBlock")

class CROSS_TRANSFORMER_IMG_VSLT_SSSCCC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model_dim = args.transformer_dim
        img_size = 224
        img_dim = 32
        img_num_layers = [1, 1, 1, 1]
        self.num_features = int(img_dim * 2 ** (len(img_num_layers) - 1))
        
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

        self.vslt_input_size = len(args.vitalsign_labtest)        
        
        self.init_fc = nn.Linear(self.num_nodes + 2, self.cross_model_dim)

        if args.img_model_type == "vit":
            self.img_encoder = ViT(in_channels=1, img_size=(img_size,img_size), 
                           patch_size=args.vit_patch_size, spatial_dims=2,
                           hidden_size=256, mlp_dim=1024,
                           num_heads = 8, num_layers=args.vit_num_layers, 
                           classification=False, num_classes=1)
        elif args.img_model_type == "swin":
            self.img_encoder = SWIN_TRANSFORMER_MODEL(
                            img_size=img_size, patch_size=4, in_chans=1, num_classes=1,
                            embed_dim=img_dim, depths=img_num_layers, num_heads=[2, 4, 8, 16],
                            window_size=7, mlp_ratio=4., qkv_bias=True,
                            drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.,
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True, enc_bool=True
                        )
        elif args.img_model_type == "resnet":
            self.img_encoder = RESNET(block=RESNET_BLOCK, layers=[2, 2, 2, 2], block_inplanes=[32,64,128,256],
                        spatial_dims=2, conv1_t_stride=2, n_input_channels=1, num_classes=1)
        else:
            raise NotImplementedError('vit or Swin Model Needed')


        self.self_attn_encoder = TransformerEncoder(
            d_input = self.self_model_dim,
            n_layers = self.self_num_layers,
            n_head = self.self_num_heads,
            d_model = self.self_model_dim,
            d_ff = self.self_model_dim * 2,
            dropout = self.self_dropout,
            pe_maxlen = 500,
            use_pe = True,
            classification = False
        )
        
        self.cross_attn_encoder = CrossTransformerEncoder(
            d_input = self.cross_model_dim,
            n_layers = self.cross_num_layers,
            n_head = self.cross_num_heads,
            d_model = self.cross_model_dim,
            d_ff = self.cross_model_dim * 2,
            dropout = self.cross_dropout,
            pe_maxlen = 500,
            use_pe = False,
            classification = False,
            mask = None
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
        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, self.self_model_dim))

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, img):
        ### IMG transformer ###
        if self.args.img_model_type == "vit":
            img_output, _ = self.img_encoder(img)
        elif self.args.img_model_type == "swin":
            img_output = self.img_encoder(img)
        elif self.args.img_model_type == "resnet":
            img_output = self.img_encoder(img)
        else:
            raise NotImplementedError('vit or Swin Model Needed')

        ### VSLT encoder ###
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        emb = self.init_fc(x)
        cls_tokens = self.cls_tokens.repeat(emb.size(0), 1, 1)
        emb = torch.cat([cls_tokens, emb], axis=1)
        
        # print("#####self#####")
        self_enc_output, _ = self.self_attn_encoder(emb,
                                           input_lengths = input_lengths)
        # print("#####cross#####")
        cross_enc_output, _ = self.cross_attn_encoder(padded_q_input = self_enc_output, 
                                            padded_kv_input = img_output, 
                                            q_input_lengths = self_enc_output.size(1),
                                            kv_input_lengths = None)
        finalClsOutput = cross_enc_output[:, 0]
        
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(finalClsOutput))
        output = torch.stack(multitask_vectors)
        output = self.sigmoid(output)

        return output