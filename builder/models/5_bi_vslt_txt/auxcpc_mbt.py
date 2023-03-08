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

class AUXCPC_MBT(nn.Module):
    def __init__(self, args):

        super(AUXCPC_MBT, self).__init__()
        
        self.n_modality = 2

        self.num_layers = 8
        self.num_heads = args.transformer_num_head
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
            
        self.encoder = nn.Sequential(
                            nn.Linear(self.num_nodes+2, self.model_dim),
                            nn.LayerNorm(self.model_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.model_dim, self.model_dim),
                            nn.LayerNorm(self.model_dim),
                            nn.ReLU(inplace=True),
                            )
        
        self.c_t_Model = MBTEncoder(
            n_modality = self.n_modality,
            bottlenecks_n = 4,      # https://arxiv.org/pdf/2107.00135.pdf # according to section 4.2 implementation details
            fusion_startidx = args.mbt_fusion_startIdx,
            d_input = self.model_dim,
            n_layers = self.num_layers,
            n_head = self.num_heads,
            d_model = self.model_dim,
            d_ff = self.model_dim * 4,
            dropout = self.dropout,
            pe_maxlen = 600,
            use_pe = [True, True],
            mask = [True, True],
        )
        self.layer_norms_after_concat = nn.LayerNorm(self.model_dim * 2)
        
        # self.Wk  = nn.ModuleList([nn.Linear(512, 256) for i in range(self.timestep)])
        self.ct_weight = nn.Parameter(torch.zeros((12,512,256)), requires_grad=True)
        torch.nn.init.uniform_(self.ct_weight, a=-math.sqrt(512), b=math.sqrt(512))
        
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
            nn.Linear(in_features=self.model_dim * 2, out_features= 64, bias=True),
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

    def forward(self, x, f_indices, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        loss = 0 # average over timestep and batch
        batch = x.size()[0]
        t_samples = 24
        # input sequence is N*C*L
        
        ############################### Encoder ################################
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        z = self.encoder(x)
        if x.size(1) > t_samples:
            if self.neg_sample_from == "Future":        
                f_indices = f_indices[:,-12:].contiguous().view(-1)
                mask = f_indices.expand(12*batch,-1).ne(1)
                enc_samples = z[:,t_samples:t_samples+12,:].contiguous()
                z = z[:,:t_samples,:] # e.g. size N*100*512
            else:
                enc_samples = z[:,:,:]
                z = z[:,:t_samples,:] # e.g. size N*100*512
        # print("mask: ", mask)
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        output, _ = self.c_t_Model([z, txt_embedding], 
                                      lengths = [input_lengths, txt_lengths + 2])
        
        ################################# G_ar #################################
        c_t = torch.cat([output[i][:, 0, :] for i in range(self.n_modality)], dim=1)
        c_t = self.layer_norms_after_concat(c_t)

        ############################ Auxiliary Loss ############################
        if x.size(1) > t_samples:
            # pred = N x 12 x 256
            # enc_samples = N x 12 x 256
            pred = torch.transpose(torch.matmul(c_t, self.ct_weight), 0, 1).contiguous()
            if "Wbrl" in self.use_BRL:
                pred = self.B(pred.permute(0,2,1))
                pred = self.RL(pred.permute(0,2,1))
                
            if self.aux_loss == 1:              # cpcWbrl (0~10)
                total = torch.mm(pred.view(-1,256), torch.transpose(enc_samples.view(-1,256), 0, 1))
                total.masked_fill_(mask, -65504)
                total.masked_fill_(self.ssample_mask, -65504)
                loss = torch.sum(torch.log((torch.diag(self.softmax(total)) * f_indices) + 1e-7) * f_indices)
                # nce /= -1.*batch*self.timestep
                loss /= -1.* torch.sum(f_indices)
            elif self.aux_loss == 2:            # very small on both cosineWbrl and cosine (-1~1)
                loss = -(torch.sum(self.cosine_loss(pred.view(-1,256), enc_samples.view(-1,256)) * f_indices) / torch.sum(f_indices))
            elif self.aux_loss == 3:            # l2 very high (ex.86000) and good enough for l2Wbrl (ex.1.5)
                loss = torch.sum(torch.mean(self.l2_loss(pred.view(-1,256), enc_samples.view(-1,256)), dim=1) * f_indices)
                loss /= torch.sum(f_indices)
        ############################ Supervised Loss ############################
        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(c_t))
        output = torch.stack(multitask_vectors)
        # print(output.shape)
        # print("loss: ", loss.shape)
        # print("loss: ", loss)
        # exit(1)
        return output, loss
