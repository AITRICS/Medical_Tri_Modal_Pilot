import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

# early fusion

class MMVED(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size
        self.device = args.device
        self.classifier_nodes = 64
        self.hidden_size = args.hidden_size
        self.img_size = 224
        img_hidden_size = 256
        patch_size = 16
        img_num_heads = 4
        pos_embed = "conv"
        
        #Image
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=1,
            img_size=self.img_size,
            patch_size=patch_size,
            hidden_size=img_hidden_size,
            num_heads=img_num_heads,
            pos_embed=pos_embed,
            dropout_rate=0,
            spatial_dims=2,
        )
        
        # Feature Extractor (for time-series)
        self.extractor_physio = nn.Sequential(nn.LayerNorm([24,18]),
                                              nn.Linear(18,256)).to(self.args.device)

        ##### Section 1: Intra & Inter view #####
        ### intra layer ###
        self.intra_physio_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        self.intra_text_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        self.intra_img_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        ### inter layer ###
        self.inter_forward_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=False, batch_first=True)
        self.inter_backward_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=False, batch_first=True)
        
        ### Tri View ###
        self.tri_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        ### Non-redundant Information Learning Layer ###        
        ### Self-Attention Layer ###        
        self.wq1_linear = nn.Linear(in_features=4608, out_features=4608, bias=False)
        self.wq1_tanh = nn.Tanh()
        self.wp1_linear = nn.Linear(in_features=4608, out_features=4608, bias=False)
        self.selfattn_softmax = torch.nn.Softmax(dim=1)
        self.wr1_linear = nn.Linear(in_features=4608, out_features=4608, bias=True)
        self.wr1_tanh = nn.Tanh()
        self.mp = nn.MaxPool1d(8, stride=4)
        
        ### Predictive Classifier Layer
        self.wp2_linear = nn.Linear(in_features=511, out_features= 128, bias=True)
        self.wp2_tanh = nn.Tanh()
        self.wq2_linear = nn.Linear(in_features=128, out_features= 1, bias=True)
        self.sigm = nn.Sigmoid()
        
        ### Mask ###
        # H_U_P, H_U_T, H_U_I, H_b1_pt, H_b2_pi, H_b3_it, H_t_P, H_t_T, H_t_I
        missing_size = [self.args.batch_size, 512, 24]
        missing_false = torch.zeros(missing_size, dtype=torch.bool)
        missing_true = torch.ones(missing_size, dtype=torch.bool)
        self.mask_img = torch.cat([missing_false,missing_false, missing_true,
                                    missing_false, missing_true, missing_true,
                                    missing_false, missing_false, missing_true], axis=1)# F,F,T, F,T,T, F,F,T
        self.mask_txt = torch.cat([missing_false, missing_true, missing_false,
                                   missing_true, missing_false, missing_true,
                                   missing_false, missing_true, missing_false], axis=1)# F,T,F, T,F,T, F,T,F
        self.mask_txt_img = torch.cat([missing_false, missing_true, missing_true,
                                       missing_true, missing_true, missing_true,
                                       missing_false, missing_true, missing_true], axis=1)# F,T,T, T,T,T, F,T,T 

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

        self.fc = nn.Sequential(
            nn.Linear(in_features=511, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True))

        self.relu = self.activations[activation]

        datasetType = args.train_data_path.split("/")[-2]
        self.txt_embedding = nn.Embedding(30000, self.hidden_size)
        
    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths, x_img, exist_img, missing_num, feasible_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths):
        # self, x, h, m, d, x_m, age, gen, length, x_txt, txt_lengths, x_img, exist_img, missing_num, feasible_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths
        # CUDA_VISIBLE_DEVICES=0 python 2_train.py --project-name icassp --input-types vslt_txt --model icassp_multiview --predict-type multi_task_range --modality-inclusion fullmodal --prediction-range 12 --lr-max 1e-4 --output-type mortality --batch-size 128 --epochs 100 --txt-tokenization bert
        modify_input_lengths = input_lengths - 1
        modify_txt_lengths = txt_lengths + 1 # SOS, EOS도 고려 후, index 추출이라
        
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        extract_x = self.extractor_physio(x)
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts) # bert: [batch,bert(128), hidden_size(256)]
        img_embedding = self.patch_embedding(x_img) # [batch, patch(196), hidden_size(256)]
        

        ### intra layer ###
        H_U_P, (hn_UP, cn_UP) = self.intra_physio_lstm(extract_x)                       # H_U_P: 128, 24, 512
        H_U_P_f = H_U_P[torch.arange(H_U_P.size(0)), modify_input_lengths, :self.hidden_size]
        H_U_P_b = H_U_P[torch.arange(H_U_P.size(0)), 0, self.hidden_size:]
        H_U_P = torch.cat([H_U_P_f, H_U_P_b], axis=1)

        H_U_T, (hn_UT, cn_UT) = self.intra_text_lstm(txt_embedding)             # H_U_T: 128, 128, 512
        H_U_T_f = H_U_T[torch.arange(H_U_T.size(0)), modify_txt_lengths, :self.hidden_size]
        H_U_T_b = H_U_T[torch.arange(H_U_T.size(0)), 0, self.hidden_size:]
        H_U_T = torch.cat([H_U_T_f, H_U_T_b], axis=1)
        
        H_U_I, (hn_UT, cn_UT) = self.intra_img_lstm(img_embedding)             # H_U_I: 128, 196, 512
        H_U_I_f = H_U_I[torch.arange(H_U_I.size(0)), exist_img*196 - 1, :self.hidden_size]
        H_U_I_b = H_U_I[torch.arange(H_U_I.size(0)), 0, self.hidden_size:]
        H_U_I = torch.cat([H_U_I_f, H_U_I_b], axis=1)
        
        
        
        ### inter layer ###
        # time-series는 forward만 2번, image는 backward만 2번
        p_packed_input = torch.nn.utils.rnn.pack_padded_sequence(extract_x, input_lengths.to("cpu"), batch_first=True, enforce_sorted=False).to(self.args.device)
        H_b1_f_P, (hf_P_1, cf_P_1) = self.inter_forward_lstm(p_packed_input)
        H_b1_f_P_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(H_b1_f_P, batch_first=True, total_length=24)
        
        t_packed_input_flip = torch.nn.utils.rnn.pack_padded_sequence(torch.flip(txt_embedding,[1]), (txt_lengths + 2).to("cpu"), batch_first=True, enforce_sorted=False).to(self.args.device)
        H_b1_b_T, (hb_T, cb_T) = self.inter_backward_lstm(t_packed_input_flip)
        H_b1_b_T_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(H_b1_b_T, batch_first=True, total_length=128)
        
        H_b2_f_P, (hf_P_2, cf_P_2) = self.inter_forward_lstm(p_packed_input)
        H_b2_f_P_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(H_b2_f_P, batch_first=True, total_length=24)
        exist_img2 = exist_img*196
        exist_img2[exist_img2==0]=1
        i_packed_input_flip = torch.nn.utils.rnn.pack_padded_sequence(torch.flip(img_embedding,[1]), exist_img2.to("cpu"), batch_first=True, enforce_sorted=False).to(self.args.device)
        H_b2_b_I, (hb_I, cb_I) = self.inter_backward_lstm(i_packed_input_flip)
        H_b2_b_I_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(H_b2_b_I, batch_first=True, total_length=196)

        t_packed_input = torch.nn.utils.rnn.pack_padded_sequence(txt_embedding, (txt_lengths + 2).to("cpu"), batch_first=True, enforce_sorted=False).to(self.args.device)
        H_b3_f_T, (hf_T, cf_T) = self.inter_forward_lstm(t_packed_input)
        H_b3_f_T_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(H_b3_f_T, batch_first=True, total_length=128)
        
        H_b3_b_I, (hb_I, cb_I) = self.inter_backward_lstm(i_packed_input_flip)
        H_b3_b_I_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(H_b3_b_I, batch_first=True, total_length=196)
        
        
        H_b1_pt = torch.cat([H_b1_f_P_unpacked[torch.arange(H_b1_f_P_unpacked.size(0)),modify_input_lengths,:], 
                             H_b1_b_T_unpacked[torch.arange(H_b1_b_T_unpacked.size(0)),0,:]], axis=1)
        H_b2_pi = torch.cat([H_b2_f_P_unpacked[torch.arange(H_b2_f_P_unpacked.size(0)),modify_input_lengths,:], 
                             H_b2_b_I_unpacked[torch.arange(H_b2_b_I_unpacked.size(0)),0,:]], axis=1)
        H_b3_it = torch.cat([H_b3_f_T_unpacked[torch.arange(H_b3_f_T_unpacked.size(0)),modify_txt_lengths,:], 
                             H_b3_b_I_unpacked[torch.arange(H_b3_b_I_unpacked.size(0)),0,:]], axis=1)
        

        
        ### Tri-View ###
        H_t_P, (hn_tP, cn_tP) = self.tri_lstm(extract_x)                       # H_U_P: 128, 24, 512
        H_t_P_f = H_t_P[torch.arange(H_t_P.size(0)), modify_input_lengths, :self.hidden_size]
        H_t_P_b = H_t_P[torch.arange(H_t_P.size(0)), 0, self.hidden_size:]
        H_t_P = torch.cat([H_t_P_f, H_t_P_b], axis=1)
        
        H_t_T, (hn_tT, cn_tT) = self.tri_lstm(txt_embedding)             # H_U_T: 128, 128, 512
        H_t_T_f = H_t_T[torch.arange(H_t_T.size(0)), modify_txt_lengths, :self.hidden_size]
        H_t_T_b = H_t_T[torch.arange(H_t_T.size(0)), 0, self.hidden_size:]
        H_t_T = torch.cat([H_t_T_f, H_t_T_b], axis=1)
        
        H_t_I, (hn_tT, cn_tT) = self.tri_lstm(img_embedding)             # H_U_I: 128, 196, 512
        H_t_I_f = H_t_I[torch.arange(H_t_I.size(0)), exist_img*196 - 1, :self.hidden_size]
        H_t_I_b = H_t_I[torch.arange(H_t_I.size(0)), 0, self.hidden_size:]
        H_t_I = torch.cat([H_t_I_f, H_t_I_b], axis=1)
        
        
        # lstm 2개 forward, backward로 
        # bi면 dual pt만 필요 
        # 비슷한 정보를 가질 필요 없어 중요한 거 담당해서 보라고 
        # text2번 , bi면 2개만 loss 나누기 2, 
        # Cf= 512*9한 것 
        # bi t가 안들어오면 uni t랑 dual의 pt missing을 해야함 
        # transformer attention sclaed dot product init에 경우의수에 따라 MaskCNN
        
        ### Non-redundant Information Learning Layer ###       
        # loss_p_t = torch.div(torch.square(torch.inner(H_t_P, H_t_T)), x.size(0))
        loss_p_t = torch.sum(torch.diagonal(loss_p_t, 0))
        # loss_t_p = torch.div(torch.square(torch.inner(H_t_T, H_t_P)), x.size(0))
        # loss_t_i = torch.div(torch.square(torch.inner(H_t_T, H_t_I)), x.size(0))
        # loss_i_t = torch.div(torch.square(torch.inner(H_t_I, H_t_T)), x.size(0))
        # loss_p_i = torch.div(torch.square(torch.inner(H_t_P, H_t_I)), x.size(0))
        # loss_i_p = torch.div(torch.square(torch.inner(H_t_I, H_t_P)), x.size(0))
        
        aux_loss = (loss_p_t + loss_t_p + loss_t_i + loss_i_t + loss_p_i + loss_i_p) / 6
        
        ### Self-Attention Layer ###        
        C_f = torch.cat([H_U_P, H_U_T, H_U_I, H_b1_pt, H_b2_pi, H_b3_it, H_t_P, H_t_T, H_t_I], axis=1)
        C_f_self = self.wp1_linear(self.wq1_tanh(self.wq1_linear(C_f)))
        ## Missing ##
        # missing_num == 0: txt + vslt + img
        # missing_num == 2: txt + vslt
        # missing_num == 1: vslt + img
        # missing_num == 3: vslt

        if missing_num == 1:
            C_f_self.masked_fill_(self.mask_img, -65504)
        if missing_num == 2:
            C_f_self.masked_fill_(self.mask_txt, -65504)
        if missing_num == 3:
            C_f_self.masked_fill_(self.mask_txt_img, -65504)
        
        ####
        C_f_self = self.selfattn_softmax(C_f_self)
        F_f = C_f_self * C_f
        R_f = self.wr1_tanh(self.wr1_linear(F_f))
        R_f = self.mp(R_f)
        
        ### Predictive Classifier Layer        
        output = self.fc(R_f)

        return output, aux_loss