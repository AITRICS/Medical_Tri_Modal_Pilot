# https://github.com/nyuad-cai/MedFuse/blob/3159207a7f61d5920e445bd7dfc25c16b7dc0145/trainers/fusion_trainer.py#L21
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
	
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = 1 #self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = 256 #768 #self.cxr_model.feats_dim
        projection_in = 768 #self.cxr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 3 * self.ehr_model.feats_dim #full 일 때만 가능

        # self.fused_cls = nn.Sequential(
        #     nn.Linear(feats_dim, 1 ),#1: self.args.num_classes
        #     nn.Sigmoid()
        # )

        
        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            # nn.Sigmoid()
        ) 
        

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)
        
        
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
            
    
    # 
    def forward(self, x, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs_img=None, pairs_txt=None, missing=None):
        if self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        else:
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, txts = txts, txt_lengths = txt_lengths, pairs_img = pairs_img, pairs_txt = pairs_txt, missing = missing)

    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , out = self.ehr_model(x, seq_lengths)
        return out
    
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs_img=None, pairs_txt=None, missing =None):
        # txt: [batch, times, dim(768)], x(ehr): [batch, time, features], img: [batch, channel(1), image_size(224), image_size(224)]
        
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        # ehr_feats: [batch, lstm dimesnion(256)]
        
        # txt: [batch, times, dim(768)]
        if self.txt_emb_type == "biobert":
            txt_embedding = self.txtnorm(txts)
            txt_embedding = self.txt_embedding(txt_embedding)
            # txt_embedding: [batchsize, self.model_dim(256)]
        else:
            txts = txts.type(torch.IntTensor).to(self.args.device) 
            txt_embedding = self.txt_embedding(txts)
            
        txt_embedding[list(~np.array(pairs_txt))] = 0
        
        # img: [batch, channel(1), image_size(224), image_size(224)]
        cxr_feats = self.cxr_model(img).squeeze() 
        # cxr_feats: [batch, dimension(7), dimesnion(7), swin_dimension(768)]
        cxr_feats = cxr_feats.permute(0,3,1,2)
        # cxr_feats: [batch, swin_dimension(768), dimension(7), dimesnion(7)]
        cxr_feats = self.cxr_model.avgpool(cxr_feats)
        # [batch, swin_dimension(768), 1, 1]
        cxr_feats = torch.flatten(cxr_feats, 1)
        # [batch, swin_dimension(768)]
        
        cxr_feats = self.projection(cxr_feats)
        # [batch, lstm dimesnion(256)]

        cxr_feats[list(~np.array(pairs_img))] = 0
        if len(ehr_feats.shape) == 1:
            feats = ehr_feats[None,None,:]
            txt_feats = txt_embedding[:,None,:]
            feats = torch.cat([txt_feats, feats, cxr_feats[:,None,:]], dim=1)
        else:
            feats = ehr_feats[:,None,:] # [batch, 1, lstm dimesnion(256)]
            txt_feats = txt_embedding[:,None,:] # [batch, 1, self.model_dim(256)]
            feats = torch.cat([txt_feats, feats, cxr_feats[:,None,:]], dim=1) # [batch, 3(n_modality), lstm dimesnion(256)]

        new_feats = torch.zeros(self.args.batch_size, 3, self.model_dim, dtype = torch.float16).to(self.args.device)
        new_feats[missing==0] = feats[missing==0] # txt + vslt + img
        new_feats[:,0:2,:][missing==2] = feats[:,0:2,:][missing==2] # txt + vslt
        new_feats[:,0:2,:][missing==1] = feats[:,1:3,:][missing==1] # vslt + img
        new_feats[:,0,:][missing==3] = feats[:,1,:][missing==3] # vslt
        
        
        
        seq_lengths = np.array([4] * len(seq_lengths)) #(batch,) # [4]로 설정한 이유: 변화과정 보기 위한 임의의 수 
        missing = missing.to("cpu")
        seq_lengths[missing==3] = 1
        seq_lengths[missing==2] = 2
        seq_lengths[missing==1] = 2
        seq_lengths[missing==0] = 3
        
        new_feats = torch.nn.utils.rnn.pack_padded_sequence(new_feats, seq_lengths, batch_first=True, enforce_sorted=False)
        new_feats = new_feats.to(self.args.device)
        x, (ht, _) = self.lstm_fusion_layer(new_feats)
        
        # lstm_out,_=torch.nn.utils.rnn.pad_packed_sequence(x,batch_first=True)
        # lstm_output = torch.zeros(self.args.batch_size,lstm_out.shape[-1], dtype = torch.float16).to(self.args.device)
        # lstm_output[seq_lengths==3]=lstm_out[seq_lengths==3][:,2,:]
        # lstm_output[seq_lengths==2]=lstm_out[seq_lengths==2][:,1,:]
        # lstm_output[seq_lengths==1]=lstm_out[seq_lengths==1][:,0,:]
        out = ht.squeeze() # [batch, swin_dimension(768)]
        output = self.lstm_fused_cls(out) # [batch, 1]

        return output


class Fusion_img(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
	
        super(Fusion_img, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = 1 #self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = 256 #768 #self.cxr_model.feats_dim
        projection_in = 768 #self.cxr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        # feats_dim = 3 * self.ehr_model.feats_dim
        feats_dim = 2 * self.ehr_model.feats_dim #full 일 때만 가능 

        # self.fused_cls = nn.Sequential(
        #     nn.Linear(feats_dim, 1 ),#1: self.args.num_classes
        #     nn.Sigmoid()
        # )

        
        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            # nn.Sigmoid()
        ) 
        

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)
        
        
        self.model_dim = args.transformer_dim
        # # TXT encoder
        # if self.args.berttype == "biobert" and self.args.txt_tokenization == "bert":
        #     self.txt_emb_type = "biobert"
        #     self.txtnorm = nn.LayerNorm(768)
        #     self.txt_embedding = nn.Linear(768, self.model_dim)
            
        # else:
        #     self.txt_emb_type = "bert"
        #     datasetType = args.train_data_path.split("/")[-2]
        #     if datasetType == "mimic_icu": # BERT
        #         self.txt_embedding = nn.Embedding(30000, self.model_dim)
        #     elif datasetType == "sev_icu":
        #         raise NotImplementedError
            
    
    # 
    def forward(self, x, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs_img=None, pairs_txt=None, missing=None):
        if self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        else:
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, txts = txts, txt_lengths = txt_lengths, pairs_img = pairs_img, pairs_txt = pairs_txt, missing = missing)

    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , out = self.ehr_model(x, seq_lengths)
        return out
    
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs_img=None, pairs_txt=None, missing =None):
        # txt: [batch, times, dim(768)], x(ehr): [batch, time, features], img: [batch, channel(1), image_size(224), image_size(224)]
        
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        # ehr_feats: [batch, lstm dimesnion(256)]
        
        # img: [batch, channel(1), image_size(224), image_size(224)]
        cxr_feats = self.cxr_model(img).squeeze() 
        # cxr_feats: [batch, dimension(7), dimesnion(7), swin_dimension(768)]
        cxr_feats = cxr_feats.permute(0,3,1,2)
        # cxr_feats: [batch, swin_dimension(768), dimension(7), dimesnion(7)]
        cxr_feats = self.cxr_model.avgpool(cxr_feats)
        # [batch, swin_dimension(768), 1, 1]
        cxr_feats = torch.flatten(cxr_feats, 1)
        # [batch, swin_dimension(768)]
        
        cxr_feats = self.projection(cxr_feats)
        # [batch, lstm dimesnion(256)]

        cxr_feats[list(~np.array(pairs_img))] = 0
        if len(ehr_feats.shape) == 1:
            feats = ehr_feats[None,None,:]
            # txt_feats = txt_embedding[:,None,:]
            # feats = torch.cat([txt_feats, feats, cxr_feats[:,None,:]], dim=1)
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        else:
            feats = ehr_feats[:,None,:] # [batch, 1, lstm dimesnion(256)]
            # txt_feats = txt_embedding[:,None,:] # [batch, 1, self.model_dim(256)]
            # feats = torch.cat([txt_feats, feats, cxr_feats[:,None,:]], dim=1) # [batch, 3(n_modality), lstm dimesnion(256)]
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)

        new_feats = feats.to(self.args.device)        
        
        seq_lengths = np.array([1] * len(seq_lengths))
        seq_lengths[pairs_img] = 2
        new_feats = torch.nn.utils.rnn.pack_padded_sequence(new_feats, seq_lengths, batch_first=True, enforce_sorted=False)
        new_feats = new_feats.to(self.args.device)
        x, (ht, _) = self.lstm_fusion_layer(new_feats)
        
        out = ht.squeeze() # [batch, swin_dimension(768)]
        output = self.lstm_fused_cls(out) # [batch, 1]
        
        return output
