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
        lstm_out = 768 #self.cxr_model.feats_dim
        projection_in = 768 #self.cxr_model.feats_dim

        

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 3 * self.ehr_model.feats_dim #full 일 때만 가능
        # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, 1 ),#1: self.args.num_classes
            nn.Sigmoid()
        )

        
        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        ) 
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(self.lstm_fused_cls)
        

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
    def forward(self, x, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs=None):
        if self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        else:
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, txts = txts, txt_lengths = txt_lengths, pairs=pairs )

    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , feats = self.ehr_model(x, seq_lengths)
        return ehr_preds
    
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, txts = None, txt_lengths = None, pairs=None):


        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        
        if self.txt_emb_type == "biobert":
            txt_embedding = self.txtnorm(txts)
            txt_embedding = self.txt_embedding(txt_embedding) #torch.Size([[batchsize], 256])
        else:
            txts = txts.type(torch.IntTensor).to(self.args.device) 
            txt_embedding = self.txt_embedding(txts) # torch.Size([4, 128, 256])
        print("txt_embedding",txt_embedding.shape)
        cxr_feats = self.cxr_model(img).squeeze() #_, _ , cxr_feats
        print(cxr_feats.shape) #([64, 7, 7, 768])
        cxr_feats = cxr_feats.permute(0,3,1,2)
        cxr_feats = self.cxr_model.avgpool(cxr_feats)
        cxr_feats = torch.flatten(cxr_feats, 1)
        
        cxr_feats = self.projection(cxr_feats)

        cxr_feats[list(~np.array(pairs))] = 0
        if len(ehr_feats.shape) == 1:
            # print(ehr_feats.shape, cxr_feats.shape)
            # import pdb; pdb.set_trace()
            feats = ehr_feats[None,None,:]
            print(ehr_feats.shape)
            print(feats.shape)
            print(cxr_feats.shape)
            print(cxr_feats[:,None,:].shape)
            feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        else:
            feats = ehr_feats[:,None,:]
            txt_feats = txt_embedding[:,None,:]
            print(ehr_feats.shape)
            print(feats.shape)
            print("txt",txt_feats.shape)
            print(cxr_feats.shape)
            print(cxr_feats[:,None,:].shape)
            feats = torch.cat([feats, cxr_feats[:,None,:], txt_feats], dim=1)
            print("feats.shape", feats.shape)
        print("seq_lengths",seq_lengths) #torch.Size([64, 3, 256])
        seq_lengths = np.array([1] * len(seq_lengths)) # full 일 때만 가능
        seq_lengths[pairs] = 3 
        print("seq_lengths",seq_lengths)
        print("feats.shape",feats.shape)
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths.to("cpu"), batch_first=True, enforce_sorted=False).to(self.args.device)
        print("feats",feats)
        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(out))
        output = torch.stack(multitask_vectors)
        # fused_preds = self.lstm_fused_cls(out)
        print(output.shape)


        return output
    
