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
import importlib
from monai.networks.nets import ViT

swin_transformer_module = importlib.import_module("builder.models.2_uni_image.swin_transformer")
SWIN_TRANSFORMER_MODEL = getattr(swin_transformer_module, "SWIN_TRANSFORMER_MODEL")
resnet_module = importlib.import_module("builder.models.2_uni_image.resnet_enc")
RESNET = getattr(resnet_module, "ResNet")
RESNET_BLOCK = getattr(resnet_module, "ResNetBlock")

class MIDFUSION_IMG_TRANS_VSLT_GRUD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        img_size = 224
        img_dim = 32
        img_num_layers = [2, 2, 6, 2]
        self.num_features = int(img_dim * 2 ** (len(img_num_layers) - 1))
        
        self.num_layers = args.transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout

        self.final_num_layers = 4
        self.final_num_heads = args.final_num_heads
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
        self.input_decay = nn.ModuleList()
        for _ in range(self.vslt_input_size):
            self.input_decay.append(nn.Linear(1,1))
        
        self.hidden_decay = nn.Linear(self.vslt_input_size, self.model_dim)
        self.gru = nn.GRUCell(self.vslt_input_size * 2, self.model_dim, True)

        self.pos_encoding = PositionalEncoding(d_model=self.model_dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_dim)).to(self.args.device)
        self.layer_norm_in = nn.LayerNorm(self.model_dim)

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


        self.final_encoder = BimodalTransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.final_num_layers,
            n_head = self.final_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 600,
            use_pe = False,
            classification = True,
            mask = True
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

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, img):
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
        vslt_embedding = torch.stack(grud_output, dim=1)
        vslt_embedding = self.layer_norm_in(vslt_embedding) + self.pos_encoding(vslt_embedding.size(1))
        
        ### IMG transformer ###
        if self.args.img_model_type == "vit":
            img_output, _ = self.img_encoder(img)
        elif self.args.img_model_type == "swin":
            img_output = self.img_encoder(img)
        elif self.args.img_model_type == "resnet":
            img_output = self.img_encoder(img)
        else:
            raise NotImplementedError('vit or Swin Model Needed')

        final_input = torch.cat((vslt_embedding, img_output), dim=1)
        final_output, _ = self.final_encoder(final_input,
            first_size = vslt_embedding.size(1),
            first_lengths = input_lengths + 1,
            second_lengths = torch.tensor([img_output.size(1)]*img_output.size(0)) + 2
        )
        
        finalClsOutput = final_output[:, 0]
        
        ageTensor = age.unsqueeze(1)
        genTensor = gen.unsqueeze(1)

        classInput = torch.cat((finalClsOutput, ageTensor, genTensor), 1)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)

        output = self.sigmoid(output)

        return output