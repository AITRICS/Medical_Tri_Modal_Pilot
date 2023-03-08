# referenced to https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import math
from builder.models.src.transformer.utils import *
from builder.models.src.transformer import *
from builder.models.src.transformer.mbt_encoder import MBTEncoder
from builder.models.src.transformer.module import LayerNorm

class AUXCPC_GRUDTXT(nn.Module):
    def __init__(self, args):

        super(AUXCPC_GRUDTXT, self).__init__()
        self.args = args
        self.auxiliary_loss_type = args.auxiliary_loss_type
        self.txt_num_layers = args.transformer_num_layers
        self.txt_num_heads = args.transformer_num_head
        self.txt_dropout = args.txt_dropout
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout

        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size

        self.device = args.device
        self.classifier_nodes = 64

        self.batch_size = args.batch_size
        self.timestep = 12
        
        self.trainer = args.trainer
        
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
        
        datasetType = args.train_data_path.split("/")[-2]
        if args.txt_tokenization == "word":
            if datasetType == "mimic_icu":
                self.txt_embedding = nn.Embedding(1620, self.model_dim)
            elif datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(3720, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(45282, self.model_dim)
        elif args.txt_tokenization == "character":
            if datasetType == "mimic_icu" or datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(42, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(1130, self.model_dim)
        elif args.txt_tokenization == "bpe":
            if datasetType == "mimic_icu":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
            elif datasetType == "mimic_ed":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
            elif datasetType == "sev_icu":
                self.txt_embedding = nn.Embedding(8005, self.model_dim)
        elif args.txt_tokenization == "bert":
            if datasetType == "mimic_icu":
                self.txt_embedding = nn.Embedding(30000, self.model_dim)
            elif datasetType == "sev_icu":
                raise NotImplementedError
            
            
        self.vslt_input_size = len(args.vitalsign_labtest)
        self.input_decay = nn.ModuleList()
        for _ in range(self.vslt_input_size):
            self.input_decay.append(nn.Linear(1,1))
        
        self.hidden_decay = nn.Linear(self.vslt_input_size, self.model_dim)
        self.gru = nn.GRUCell(self.vslt_input_size * 2, self.model_dim, True)

        self.pos_encoding = PositionalEncoding(d_model=self.model_dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_dim)).to(self.args.device)
        self.layer_norm_in = nn.LayerNorm(self.model_dim)

        self.txt_encoder = TransformerEncoder(
            d_input = self.model_dim,
            n_layers = self.txt_num_layers,
            n_head = self.txt_num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 2,
            dropout = self.txt_dropout,
            pe_maxlen = 500,
            use_pe = True,
            classification = False,
            mask = True
        )
        

        self.relu = self.activations[activation]
    
        self.layer_norms_after_concat = nn.LayerNorm(self.model_dim * 2 + 2)
        
        self.ct_weight = nn.Parameter(torch.zeros((514,256)), requires_grad=True)
        torch.nn.init.uniform_(self.ct_weight, a=-math.sqrt(514), b=math.sqrt(514))
        
        self.use_BRL = args.auxiliary_loss_type
        if "Wbrl" in self.use_BRL:
            self.B = nn.BatchNorm1d(256)
            self.RL = nn.Sequential(
            self.activations[activation],
            nn.Linear(in_features=256, out_features=256,  bias=True))
        
        # self.lsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.neg_sample_from = args.neg_samples_from
        
        if "cpc" in self.use_BRL:
            self.aux_loss = 1
            print("Preparing Contrastive Loss...")
            if self.neg_sample_from == "Future":
                ssample_mask = torch.zeros([self.timestep * args.batch_size, self.timestep * args.batch_size])
                ssample_mask_element = torch.ones([self.timestep, self.timestep])
                ssample_mask_element.fill_diagonal_(0)
                for idx in range(args.batch_size):
                    ssample_mask[idx*self.timestep:idx*self.timestep+self.timestep,idx*self.timestep:idx*self.timestep+self.timestep] = ssample_mask_element
                self.ssample_mask = ssample_mask.ne(0).cuda()
        elif "cosine" in self.use_BRL:
            self.aux_loss = 2
            print("Preparing Cosine similarity loss...")
            self.cosine_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        elif "l2" in self.use_BRL:
            self.aux_loss = 3
            print("Preparing L2 loss...")
            self.l2_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
        else:
            exit(1)
            
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim * 2 + 2, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x1, h, m1, d1, x_m, age, gen, input_lengths, txts, txt_lengths, f_indices):
        # print("txts: ", txts )
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        txt_output, _ = self.txt_encoder(txt_embedding, input_lengths = txt_lengths + 2)
        txt_cls_output = txt_output[:, 0]

        loss = 0 # average over timestep and batch
        t_samples = 24
        # input sequence is N*C*L
        
        ############################### Encoder ################################
        x = x1[:,:24,:]
        m = m1[:,:24,:]
        d = d1[:,:24,:]
        x2 = x1[:,24:,:]
        m2 = m1[:,24:,:]
        d2 = d1[:,24:,:]
        h1 = h
        h2 = h
        x_d1 = []
        x_d2 = []
        for i in range(x.shape[-1]):
            x_d1.append(self.input_decay[i](d[:, :, i].unsqueeze(-1)))
            if x1.size(1) > t_samples:
                x_d2.append(self.input_decay[i](d2[:, :, i].unsqueeze(-1)))
        x_d1 = torch.cat(x_d1, dim=-1)
        x_d1 = torch.exp(-self.relu(x_d1))
        x_m = x_m.view(1, 1, -1)
        if x1.size(1) > t_samples:
            x_d2 = torch.cat(x_d2, dim=-1)
            x_d2 = torch.exp(-self.relu(x_d2))

        x = m * x + (1 - m) * x_d1 * x + (1 - m) * (1 - x_d1) * x_m
        
        if x1.size(1) > t_samples:
            x2 = m2 * x2 + (1 - m2) * x_d2 * x2 + (1 - m2) * (1 - x_d2) * x_m

        grud_output = []
        grud_output2 = []
        
        for i in range(x.size(1)):
            h_d = self.hidden_decay(d[:, i])
            h_d = torch.exp(-self.relu(h_d))
            h1 = h_d * h1
            x_t = torch.cat((x[:, i], m[:, i]), dim=-1)
            h1 = self.gru(x_t, h1)
            grud_output.append(h1)
                
        if x1.size(1) > t_samples:
            for i in range(x2.size(1)):
                h_d2 = self.hidden_decay(d2[:, i])
                h_d2 = torch.exp(-self.relu(h_d2))
                h2 = h_d2 * h2
                x_t2 = torch.cat((x2[:, i], m2[:, i]), dim=-1)
                h2 = self.gru(x_t2, h2)
                grud_output2.append(h2)
            
        vslt_embedding_raw = torch.stack(grud_output, dim=1) # torch.Size([128, 36, 256])
        vslt_embedding = vslt_embedding_raw[torch.arange(x.shape[0]), input_lengths-1] # torch.Size([128, 256])
        
        ############################### Future input shared Encoder ################################
        if x1.size(1) > t_samples:
            vslt_embedding_raw_future = torch.stack(grud_output2, dim=1) # torch.Size([128, 36, 256])
            enc_samples = vslt_embedding_raw_future[:, -1, :] # torch.Size([128, 256])
        
        ageTensor = age.unsqueeze(1)
        genTensor = gen.unsqueeze(1)
        
        c_t = torch.cat((vslt_embedding, txt_cls_output, ageTensor, genTensor), 1)
        
        ################################# G_ar #################################
        # if self.auxiliary_loss_type == "l2":
        #     c_t = self.layer_norms_after_concat(c_t)
            
        ############################ Auxiliary Loss ############################
        if x1.size(1) > t_samples:
            # pred = N x 12 x 256
            # enc_samples = N x 12 x 256
            pred = torch.matmul(c_t, self.ct_weight).contiguous()
               
            if self.aux_loss == 1:              # cpcWbrl (0~10)
                total = torch.mm(pred, torch.transpose(enc_samples, 0, 1))
                loss = torch.mean(torch.log((torch.diag(self.softmax(total))) + 1e-7))
                loss /= -1.
            elif self.aux_loss == 2:            # very small on both cosineWbrl and cosine (-1~1)
                loss = -torch.mean(self.cosine_loss(pred, enc_samples))
            elif self.aux_loss == 3:            # l2 very high (ex.86000) and good enough for l2Wbrl (ex.1.5)
                loss = torch.mean(self.l2_loss(pred, enc_samples))
        ############################ Supervised Loss ############################
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(c_t))
        output = torch.stack(multitask_vectors)
        # print(output.shape)
        # exit(1)
        return output, loss
