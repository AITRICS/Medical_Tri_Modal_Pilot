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
        dim = dim_visual + dim_ehr + args.transformer_dim # txt_embeidding 
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_txt = nn.Linear(dim_out, args.transformer_dim) 
        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_ehr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, txt_embedding, skeleton, visual):
        # txt_embedding, skeleton, visual: [batch, self.model_dim(256)], [batch, time, lstm dimension(256)], [batch, swin_dimension(768), dimension(7), dimesnion(7)]
        squeeze_array = []
        squeeze_array.append(txt_embedding) 
        ehr_avg = torch.mean(skeleton, dim=1)
        squeeze_array.append(ehr_avg) #squeeze_array[0]: [batch, self.model_dim(256)], squeeze_array[1]: [batch, swin dimension(768)], squeeze_array[2]: [batch, lstm dimension(256)]
        visual_view = visual.view(visual.shape[:2] + (-1,)) # torch.Size([batchsize, swin dimension(768), 7, 7]) -> torch.Size([batchsize, swin dimension(768), 49])
        squeeze_array.append(torch.mean(visual_view, dim=-1))
        
        squeeze = torch.cat(squeeze_array, 1) # squeeze: [batch, self.model_dim + swin dimension + lstm dimension(1280)]
        excitation = self.fc_squeeze(squeeze) # [batch, dim_out( = (dim_visual + dim_ehr + args.transformer_dim)/2 ex.640)]
        excitation = self.relu(excitation)
        
        txt_out = self.fc_txt(excitation) # [batch, self.model_dim(256)]
        sk_out = self.fc_skeleton(excitation) # [batch, lstm dimension(256)]
        vis_out = self.fc_visual(excitation) # [batch, swin_dimension(768)]

        txt_out = self.sigmoid(txt_out)
        sk_out = self.sigmoid(sk_out)
        vis_out = self.sigmoid(vis_out)
        
        dim_diff = len(txt_embedding.shape) - len(txt_out.shape)
        txt_out = txt_out.view(txt_out.shape + (1,) * dim_diff) # [batch, self.model_dim(256)]

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape[0], 1 , sk_out.shape[1]) # [batch, 1, lstm dimension(256)]

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff) # [batch, swin_dimension(768), 1, 1]

        # [batch, self.model_dim(256)], [batch, swin_dimension(768), 7, 7], [batch, times(24), lstm dimension(256)])
        return txt_embedding * txt_out, skeleton * sk_out, visual * vis_out


class FusionMMTM(nn.Module):

    def __init__(self, args, ehr_model, cxr_model):
        
        super(FusionMMTM, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.model_dim = args.transformer_dim

        self.mmtm4 = MMTM(768, self.ehr_model.feats_dim, self.args.mmtm_ratio)

        feats_dim = 3 * self.ehr_model.feats_dim #self.cxr_model.feats_dim #2를 3으로 바꿈
        

        self.joint_cls = nn.Sequential(
            nn.Linear(feats_dim, 1),#self.args.num_classes
        )
        
        self.projection_txt = nn.Linear(self.model_dim, self.cxr_model.feats_dim)
        self.classifier = nn.Sequential(nn.Linear(self.cxr_model.feats_dim, 1))
        self.projection = nn.Linear(self.ehr_model.feats_dim, self.cxr_model.feats_dim)
        self.projection_cxr = nn.Linear(self.cxr_model.feats_dim, self.ehr_model.feats_dim)
        
        
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
        
        # txt: [batch, times, self.model_dim(256)]
        if self.txt_emb_type == "biobert":
            txt_embedding = self.txtnorm(txts)
            txt_embedding = self.txt_embedding(txt_embedding) 
            # txt_embedding: [batchsize, self.model_dim(256)]
        else:
            txts = txts.type(torch.IntTensor).to(self.args.device) 
            txt_embedding = self.txt_embedding(txts)
            
        # ehr: [batch, time, features]
        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False) #padding 연산안되도록
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
        
        #[batch, self.model_dim(256)], [batch, swin_dimension(768), dimension(7), dimesnion(7)], [batch, time, lstm dimension(256)]
        txt_embedding, ehr_unpacked, cxr_feats = self.mmtm4(txt_embedding, ehr_unpacked, cxr_feats)

        ehr = torch.nn.utils.rnn.pack_padded_sequence(ehr_unpacked, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        ehr = ehr.to(self.args.device)
        ehr, (ht, _)= self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze() # ehr_feats: [batch, lstm dimension(256)]    
        ehr_feats = self.ehr_model.do(ehr_feats)


        # projected_txt = self.projection_txt(txt_embedding) # [batch, swin_dimension(768)]
        # projected = self.projection(ehr_feats) # [batch, swin_dimension(768)]
        
        cxr_feats = self.cxr_model.avgpool(cxr_feats) # [batch, swin_dimension(768), 1, 1]
        cxr_feats = torch.flatten(cxr_feats, 1) # [batch, swin_dimension(768)]
        
        projected_cxr = self.projection_cxr(cxr_feats) # [batch, lstm dimension(256)]
        
        # feats = torch.cat([projected_txt, projected, cxr_feats], dim=1) # [batch, 3*swin_dimension(3*768)]
        feats = torch.cat([txt_embedding, ehr_feats, projected_cxr]) # [batch, 3*swin_dimension(3*256)]
        
        joint_preds = self.joint_cls(feats) # [batch, 1]

        # output = torch.sigmoid(joint_preds)

        return joint_preds 
        