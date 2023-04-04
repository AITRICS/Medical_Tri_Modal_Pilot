import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *

# early fusion

class BI_MMVED(nn.Module): #text 가 맞나? IMG가 아니라
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size
        self.device = args.device
        self.classifier_nodes = 64
        self.hidden_size = args.hidden_size

        ##### Section 1: Intra & Inter view #####
        ### intra layer ###
        self.intra_physio_lstm = nn.LSTM(input_size=18, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        self.intra_text_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        
        ### inter layer ###
        self.inter_physio_forward_lstm = nn.LSTM(input_size=18, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=False, batch_first=True)
        self.inter_physio_backward_lstm = nn.LSTM(input_size=18, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=False, batch_first=True)
        self.inter_text_forward_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=False, batch_first=True)
        self.inter_text_backward_lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=False, batch_first=True)

        ### Non-redundant Information Learning Layer ###        
        ### Self-Attention Layer ###        
        self.wq1_linear = nn.Linear(in_features=2048, out_features=2048, bias=False)
        self.wq1_tanh = nn.Tanh()
        self.wp1_linear = nn.Linear(in_features=2048, out_features=2048, bias=False)
        self.selfattn_softmax = torch.nn.Softmax(dim=1)
        self.wr1_linear = nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.wr1_tanh = nn.Tanh()
        self.mp = nn.MaxPool1d(8, stride=4)
        
        ### Predictive Classifier Layer
        self.wp2_linear = nn.Linear(in_features=511, out_features= 128, bias=True)
        self.wp2_tanh = nn.Tanh()
        self.wq2_linear = nn.Linear(in_features=128, out_features= 1, bias=True)
        self.sigm = nn.Sigmoid()

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

        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=511, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        self.relu = self.activations[activation]

        datasetType = args.train_data_path.split("/")[-2]
        self.txt_embedding = nn.Embedding(30000, self.hidden_size)
        
    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        # CUDA_VISIBLE_DEVICES=0 python 2_train.py --project-name icassp --input-types vslt_txt --model icassp_multiview --predict-type multi_task_range --modality-inclusion fullmodal --prediction-range 12 --lr-max 1e-4 --output-type mortality --batch-size 128 --epochs 100 --txt-tokenization bert
        input_lengths = input_lengths - 1 
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)

        ### intra layer ###
        H_U_P, (hn_UP, cn_UP) = self.intra_physio_lstm(x)                       # H_U_P: 128, 24, 512
        H_U_P_f = H_U_P[torch.arange(H_U_P.size(0)), input_lengths, :self.hidden_size]
        H_U_P_b = H_U_P[torch.arange(H_U_P.size(0)), 0, self.hidden_size:]
        H_U_P = torch.cat([H_U_P_f, H_U_P_b], axis=1)

        H_U_T, (hn_UT, cn_UT) = self.intra_text_lstm(txt_embedding)             # H_U_T: 128, 128, 512
        H_U_T_f = H_U_T[torch.arange(H_U_T.size(0)), txt_lengths, :self.hidden_size]
        H_U_T_b = H_U_T[torch.arange(H_U_T.size(0)), 0, self.hidden_size:]
        H_U_T = torch.cat([H_U_T_f, H_U_T_b], axis=1)
        
        ### inter layer ###
        H_b1_f_P, (hf_P, cf_P) = self.inter_physio_forward_lstm(x)              # H_b1_f_P: 128, 24, 256
        H_b1_f_P = H_b1_f_P[torch.arange(H_b1_f_P.size(0)), input_lengths, :]
        
        H_b1_b_P, (hb_P, cb_P) = self.inter_physio_backward_lstm(torch.flip(x,[1]))             # H_b1_b_P: 128, 24, 256
        H_b1_b_P = H_b1_b_P[torch.arange(H_b1_b_P.size(0)), -1, :]
        
        H_b2_f_T, (hf_T, cf_T) = self.inter_text_forward_lstm(txt_embedding)    # H_b2_f_T: 128, 128, 256
        H_b2_f_T = H_b2_f_T[torch.arange(H_b2_f_T.size(0)), txt_lengths, :]
        
        H_b2_b_T, (hb_T, cb_T) = self.inter_text_backward_lstm(torch.flip(txt_embedding,[1]))   # H_b2_b_T: 128, 128, 256
        H_b2_b_T = H_b2_b_T[torch.arange(H_b2_b_T.size(0)), -1, :]
        
        H_b_P = torch.cat([H_b1_f_P, H_b2_b_T], axis=1)
        H_b_T = torch.cat([H_b2_f_T, H_b1_b_P], axis=1)
        
        ### Non-redundant Information Learning Layer ###       
        loss_p = torch.div(torch.square(torch.inner(H_b_P, H_b_T)), x.size(0))
        loss_t = torch.div(torch.square(torch.inner(H_b_T, H_b_P)), x.size(0))
        
        ### Self-Attention Layer ###        
        C_f = torch.cat([H_U_P, H_U_T, H_b_P, H_b_T], axis=1)
        C_f_self = self.wp1_linear(self.wq1_tanh(self.wq1_linear(C_f)))
        C_f_self = self.selfattn_softmax(C_f_self)
        F_f = C_f_self * C_f
        R_f = self.wr1_tanh(self.wr1_linear(F_f))
        R_f = self.mp(R_f)
        
        ### Predictive Classifier Layer
        # output = self.sigm(self.wq2_linear(self.wp2_tanh(self.wp2_linear(R_f))))
        
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(R_f))
        output = torch.stack(multitask_vectors)

        # print("output: ", output.shape) # torch.Size([12, batch_size, 1])

        return output