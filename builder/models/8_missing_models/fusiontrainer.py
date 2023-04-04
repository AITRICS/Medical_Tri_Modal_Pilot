#https://github.com/nyuad-cai/MedFuse/blob/3159207a7f61d5920e445bd7dfc25c16b7dc0145/trainers/fusion_trainer.py#L55

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from builder.models.src.baseline_medfuse import Fusion, Fusion_img
from builder.models.src.baseline_mmtm import FusionMMTM
from builder.models.src.baseline_daft import FusionDAFT
# from models.ehr_models import LSTM
# from models.cxr_models import CXRModels
# from .trainer import Trainer
import pandas as pd
from control.config import args
from builder.models.src.lstm import LSTM
# from builder.models.src.resnet import resnet34, resnet18, ResNet34_Weights, ResNet18_Weights
from builder.utils.cosine_annealing_with_warmup_v2 import CosineAnnealingWarmupRestarts
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle
from builder.models.src.swin_transformer import swin_t_m, Swin_T_Weights

import numpy as np
from sklearn import metrics
               
class FUSIONTRAINER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_classes = 1
        self.dim = 256
        self.dropout = 0.3
        self.layers = 2
        
        self.ehr_model = LSTM(input_dim=18, num_classes=self.num_classes, hidden_dim=self.dim, dropout=self.dropout, layers=self.layers).to(self.device)
        self.cxr_model = swin_t_m(weights = Swin_T_Weights.IMAGENET1K_V1).to(args.device)#Swin_T_Weights.IMAGENET1K_V1

        models = [self.ehr_model, self.cxr_model]
        model_dict = models[0].state_dict()
        if self.args.output_type == "mortality":
            dir = "/mnt/aitrics_ext/ext01/claire/multimodal/mlhc_pretrained_model/ehr_image_reports_Mortal_swin_1e-4_0328/best_fold2_seed2023.pth"
        elif self.args.output_type == "vasso":
            dir = "/mnt/aitrics_ext/ext01/claire/multimodal/mlhc_pretrained_model/ehr_image_reports_Intub_swin_1e-4_0328/best_fold1_seed1004.pth"
        elif self.args.output_type =="intubation":
            dir = "/mnt/aitrics_ext/ext01/claire/multimodal/mlhc_pretrained_model/ehr_image_reports_Vasso_swin_1e-4_0328/best_fold0_seed412.pth"
        old_weights=torch.load(dir)['model']
        new_weights=torch.load(dir)['model']
        new_weights = {key.replace('ehr_model.', ''): new_weights.pop(key) for key in old_weights.keys()}
        new_weights = {k: v for k, v in new_weights.items() if k in model_dict}
        model_dict.update(new_weights)
        models[0].load_state_dict(new_weights)

        model_dict = models[1].state_dict()
        old_weights=torch.load("/mnt/aitrics_ext/ext01/claire/multimodal/mlhc_pretrained_model/image_reports_swin_1e-6_resize_affine_crop-resize_crop_0323/best_fold0_seed0.pth")['model']
        new_weights=torch.load("/mnt/aitrics_ext/ext01/claire/multimodal/mlhc_pretrained_model/image_reports_swin_1e-6_resize_affine_crop-resize_crop_0323/best_fold0_seed0.pth")['model']
        new_weights = {key.replace('img_encoder.', ''): new_weights.pop(key) for key in old_weights.keys()}
        new_weights = {k: v for k, v in new_weights.items() if k in model_dict}
        model_dict.update(new_weights)
        models[1].load_state_dict(new_weights)

        self.cxr_model.feats_dim = 768 #swin 512 #resnet34  
        
        if 'uni_ehr' in self.args.fusion_type:
            for p in self.cxr_model.parameters():
                p.requires_grad = False

        if args.fuse_baseline == "Medfuse":
            if self.args.input_types =="vslt_img":
                self.model = Fusion_img(args, self.ehr_model, self.cxr_model ).to(self.device)
            else:
                self.model = Fusion(args, self.ehr_model, self.cxr_model ).to(self.device)
        elif args.fuse_baseline == "MMTM":
            self.model = FusionMMTM(args, self.ehr_model, self.cxr_model ).to(self.device)
        elif args.fuse_baseline == "DAFT":
            self.model = FusionDAFT(args, self.ehr_model, self.cxr_model ).to(self.device)
        else:
            print('choose one of ["Medfuse", "MMTM","DAFT","Retain","Multi"]')
            exit(1)



        # self.pairs = [True for item in range(args.batch_size)]# full model만 가능 
        
    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, img, exist_img, missing, f_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths):
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        pairs_txt = torch.tensor(txt_lengths.to("cpu"), dtype=bool)
        pairs_img = torch.tensor(exist_img.to("cpu"), dtype=bool)

        final_output = self.model(x, input_lengths, img, txts, txt_lengths, pairs_img, pairs_txt, missing)
        # input_lengths: list of sequence lengths of each batch element

        return final_output, None