# https://github.com/nyuad-cai/MedFuse/blob/3159207a7f61d5920e445bd7dfc25c16b7dc0145/models/fusion_mmtm.py#L54
import torch.nn as nn
import torchvision
import torch
import numpy as np
import torch.optim as optim
from torch.nn.functional import kl_div, softmax, log_softmax
# from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet
from control.config import args
import numpy as np

class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_ehr, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_ehr
        dim_out = int(2*dim/ratio)
        print("dim",dim)
        print(dim_out)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_ehr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, visual, skeleton):
        squeeze_array = []
        visual_view = visual.view(visual.shape[:2] + (-1,))
        squeeze_array.append(torch.mean(visual_view, dim=-1))
        # print(squeeze_array[0].shape,"eko")
        # print(visual_view.shape)
        
        ehr_avg = torch.mean(skeleton, dim=1)
        print(ehr_avg.shape)

        squeeze_array.append(ehr_avg)

        squeeze = torch.cat(squeeze_array, 1)
        print("squeeze",squeeze.shape)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        # print("dun_diff",dim_diff)
        # print("fc",vis_out.shape)
        # print("fc",sk_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)
        # print(vis_out.shape)
        # print(vis_out)
        # print("outpput",(visual * vis_out).shape)


        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        # print("dun_diff",dim_diff)
        sk_out = sk_out.view(sk_out.shape[0], 1 , sk_out.shape[1])
        # print(sk_out)
        # print(sk_out.shape)
        # print("outpput",(skeleton * sk_out).shape)
        return visual * vis_out, skeleton * sk_out


class FusionMMTM(nn.Module):

    def __init__(self, args, ehr_model, cxr_model):
	
        super(FusionMMTM, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        self.mmtm0 = MMTM(64, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm1 = MMTM(64, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm2 = MMTM(128, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm3 = MMTM(256, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm4 = MMTM(768, self.ehr_model.feats_dim, self.args.mmtm_ratio)

        feats_dim = 2 * self.cxr_model.feats_dim
        

        self.joint_cls = nn.Sequential(
            nn.Linear(feats_dim, 1),#self.args.num_classes
        )

        self.classifier = nn.Sequential(nn.Linear(self.cxr_model.feats_dim, 1))
        self.projection = nn.Linear(self.ehr_model.feats_dim, self.cxr_model.feats_dim)

        # self.align_loss = CosineLoss()
        # self.kl_loss = KLDivLoss()
        
        self.fc_list_cxr = nn.ModuleList()
        for _ in range(12):
            self.fc_list_cxr.append(self.classifier)
            
        self.fc_list_ehr = nn.ModuleList()
        for _ in range(12):
            self.fc_list_ehr.append(self.ehr_model.dense_layer)
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(self.joint_cls)
        self.model_dim = args.transformer_dim
        
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

    def forward(self, ehr, seq_lengths=None, img=None, txts = None, txt_lengths = None, n_crops=0, bs=16):

        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False) #padding 연산안되도록
        ehr = ehr.to(self.args.device)
        ehr, (ht, _)= self.ehr_model.layer0(ehr)
        ehr_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(ehr, batch_first=True)
        # print(ehr_unpacked.shape)
        print("self.cxr_model.feats_dim",self.cxr_model.feats_dim)

        cxr_feats = self.cxr_model.features(img)
        cxr_feats = self.cxr_model.norm(cxr_feats)
        cxr_feats = cxr_feats.permute(0, 3, 1, 2)
        cxr_feats, ehr_unpacked = self.mmtm4(cxr_feats, ehr_unpacked)
        print(cxr_feats.shape)


        


        cxr_feats = self.cxr_model.avgpool(cxr_feats)
        print(cxr_feats.shape)
        cxr_feats = torch.flatten(cxr_feats, 1)
        # print(cxr_feats.shape)

        # cxr_preds = self.classifier(cxr_feats)
        # cxr_preds_sig = torch.sigmoid(cxr_preds)
        
        cxr_preds =[]
        cxr_preds_sig = []
        for i, fc in enumerate(self.fc_list_cxr):
            cxr_preds.append(fc(cxr_feats))
            cxr_preds_sig.append(torch.sigmoid(cxr_preds[i]))
        cxr_preds = torch.stack(cxr_preds)
        cxr_preds_sig = torch.stack(cxr_preds_sig)

        print("ehr_unpacked",ehr_unpacked.shape)
        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr_unpacked, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        ehr = ehr.to(self.args.device)
        ehr, (ht, _)= self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze()
        # print(ehr_feats.shape)
        
        ehr_feats = self.ehr_model.do(ehr_feats)
        # print(ehr_feats.shape)
        # ehr_preds = self.ehr_model.dense_layer(ehr_feats) #dense_layer도 12개 필요할 듯
        # ehr_preds_sig = torch.sigmoid(ehr_preds)
        ehr_preds=[]
        ehr_preds_sig=[]
        for i, fc in enumerate(self.fc_list_ehr):
            ehr_preds.append(fc(ehr_feats))
            ehr_preds_sig.append(torch.sigmoid(ehr_preds[i]))
        ehr_preds = torch.stack(ehr_preds)
        ehr_preds_sig = torch.stack(ehr_preds_sig)
            



        projected = self.projection(ehr_feats)
        # print(projected.shape)
        # loss = self.kl_loss(cxr_feats, projected)

        feats = torch.cat([projected, cxr_feats], dim=1)
        # print(feats.shape)
        multitask_vectors = []
        multitask_vectors2 = []
        for i, fc in enumerate(self.fc_list):
            multitask_vectors.append(torch.sigmoid(fc(feats)))
            multitask_vectors2.append(fc(feats))
        output = torch.stack(multitask_vectors)
        output2 = torch.stack(multitask_vectors2)
        
        late_average = [(cxr_preds[i] + ehr_preds[i])/2 for i in range(len(cxr_preds))]
        late_average_sig = [(cxr_preds_sig[i] + ehr_preds_sig[i])/2 for i in range(len(cxr_preds))]
        
        # joint_preds = self.joint_cls(feats)

        # joint_preds_sig = torch.sigmoid(joint_preds)


        print("output",output.shape)
       
        # exit(1)
        return {
            'cxr_only': cxr_preds_sig,
            'ehr_only': ehr_preds_sig,
            'lstm': output, #joint -> lstm
            'late_average': late_average_sig,
            # 'align_loss': loss,

            'cxr_only_scores': cxr_preds,
            'ehr_only_scores': ehr_preds,
            'late_average_scores': late_average,
            'joint_scores': output2, #joint_preds,

            }
        
