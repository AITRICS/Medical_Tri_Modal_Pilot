#https://github.com/nyuad-cai/MedFuse/blob/3159207a7f61d5920e445bd7dfc25c16b7dc0145/models/fusion_daft.py#L13
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet

class FusionDAFT(nn.Module):

    def __init__(self, args, ehr_model, cxr_model):
	
        super(FusionDAFT, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.model_dim = args.transformer_dim #256 #txt
        
        bottleneck_dim_4 = int(((4 * 4) + 768 + self.model_dim) / 7.0) # 원래 DAFT 논문 bottleneck_dim = int((4 * model_args["n_basefilters"] + self.tabular_size) / args.bottleneck_factor)

        self.daft_layer_4 = DAFTBlock(in_channels=256, ndim_non_img = 768, bottleneck_dim = bottleneck_dim_4, location = 0, activation = args.daft_activation)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_list = nn.ModuleList()
        
        # TXT encoder
        if self.args.berttype == "biobert" and self.args.txt_tokenization == "bert":
            self.txt_emb_type = "biobert"
            self.txtnorm = nn.LayerNorm(768)
            self.txt_embedding = nn.Linear(768, self.model_dim)
            
        else:
            self.txt_emb_type = "bert"
            datasetType = args.train_data_path.split("/")[-2]
            if datasetType == "mimic_icu": # BERT
                self.txt_embedding = nn.Embedding(30000, self.model_dim)
            elif datasetType == "sev_icu":
                raise NotImplementedError
            
    def forward(self, ehr, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs_txt=None, pairs_img=None):
        # txt: [batch, times, self.model_dim(256)], ehr: [batch, time, features], img: [batch, channel(1), image_size(224), image_size(224)]
        if self.txt_emb_type == "biobert":
            txt_embedding = self.txtnorm(txts)
            txt_embedding = self.txt_embedding(txt_embedding)
            # txt_embedding: [batchsize, self.model_dim(256)]
        else:
            txts = txts.type(torch.IntTensor).to(self.args.device) 
            txt_embedding = self.txt_embedding(txts)

        # ehr: [batch, time, features]
        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        ehr = ehr.to(self.args.device)
        ehr, (ht, _)= self.ehr_model.layer0(ehr)
        ehr_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(ehr, batch_first=True)
        # ehr_unpacked: [batch, time, lstm dimension(256)]
        
        # img: [batch, channel(1), image_size(224), image_size(224)]
        cxr_feats = self.cxr_model.features(img)
        # cxr_feats: [batch, dimension(7), dimesnion(7), swin_dimension(768)]
        cxr_feats = self.cxr_model.norm(cxr_feats)
        cxr_feats = cxr_feats.permute(0, 3, 1, 2)
        # cxr_feats: [batch, swin_dimension(768), dimension(7), dimesnion(7)]
        
        ehr_unpacked = self.daft_layer_4(txt_embedding, ehr_unpacked, cxr_feats)
        # [batch, times(24), lstm dimension(256)]

        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr_unpacked, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        ehr = ehr.to(self.args.device)
        ehr, (ht, _)= self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze() # ehr_feats: [batch, lstm dimension(256)]
        out = self.ehr_model.do(ehr_feats)
        out = self.ehr_model.dense_layer(out)# [batch, 1]
        # ehr_preds = torch.sigmoid(out) 

        return out
        
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class DAFTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ndim_non_img: int = 15,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 7,
    ) -> None:
        super(DAFTBlock, self).__init__()
        self.scale_activation = None
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

        self.location = location
        self.film_dims = in_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = 2 * self.film_dims # 이유: txt_embedding(256) = ehr dimension(256)
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims # ehr에 적용될 scale shift에 node 맡추기 위해서 2를 곱해준 것
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        
    def forward(self, txt_embedding, x_aux, feature_map):
        # txt_embedding, visual, skeleton: [batch, self.model_dim(256)], [batch, swin_dimension(768), dimension(7), dimesnion(7)], [batch, time, lstm dimension(256)]
        ehr_avg = torch.mean(x_aux, dim=1) # [batch, lstm dimension(256)]
        squeeze = self.global_pool(feature_map) # [batch, swin_dimension(768), 1, 1]
        squeeze = squeeze.view(squeeze.size(0), -1) # [batch, swin_dimension(768)]
        squeeze = torch.cat((txt_embedding, ehr_avg, squeeze), dim=1) # [batch, self.model_dim + swin dimension + lstm dimension(1280)]
        attention = self.aux(squeeze) #[batch, 2*lstm dimension(512)]
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1) # v_scale, v_shift: [batch, lstm dimension(256)]
            v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(x_aux)
            v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(x_aux) # v_scale, v_shift: [batch, times(24), lstm dimension(256)]
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(x_aux)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(x_aux)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )
        
        # [batch, times(24), lstm dimension(256)]
        return (v_scale * x_aux) + v_shift