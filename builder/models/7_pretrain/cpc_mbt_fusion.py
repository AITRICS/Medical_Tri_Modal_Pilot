# referenced to https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
# 
#

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

class CPC_MBT_FUSION(nn.Module):
    def __init__(self, args):

        super(CPC_MBT_FUSION, self).__init__()
        
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
        
        self.Wk  = nn.ModuleList([nn.Linear(512, 256) for i in range(self.timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
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

        # # initialize gru
        # for layer_p in self.c_t_Model._all_weights:
        #     for p in layer_p:
        #         if 'weight' in p:
        #             nn.init.kaiming_normal_(self.c_t_Model.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        nce = 0 # average over timestep and batch
        batch = x.size()[0]
        t_samples = 24
        # input sequence is N*C*L
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        z = self.encoder(x)
         
        encode_samples = torch.empty((self.timestep,batch,256)).float() # e.g. size 12*N*dim

        for i in np.arange(self.timestep):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256) # z_tk e.g. size N*dim
        forward_seq = z[:,:t_samples,:] # e.g. size N*100*512

        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        output, _ = self.c_t_Model([forward_seq, txt_embedding], 
                                      lengths = [input_lengths, txt_lengths + 2])
        c_t = torch.cat([output[i][:, 0, :] for i in range(self.n_modality)], dim=1)
        c_t = self.layer_norms_after_concat(c_t)

        pred = torch.empty((self.timestep,batch,256)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce

    def linear_eval(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        batch = x.size()[0]
        with torch.no_grad():
            z = self.encoder(x)
            txts = txts.type(torch.IntTensor).to(self.device)
            txt_embedding = self.txt_embedding(txts)
            c_t, _ = self.c_t_Model([z, txt_embedding], 
                                        lengths = [input_lengths, txt_lengths + 2])
            output = torch.cat([c_t[i][:, 0, :] for i in range(self.n_modality)], dim=1)
        
        classInput = self.layer_norms_after_concat(output)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)
        output = self.sigmoid(output)

        return output

    def prediction(self, x, h, m, d, x_m, age, gen, input_lengths, txts, txt_lengths):
        batch = x.size()[0]
        z = self.encoder(x)
        txts = txts.type(torch.IntTensor).to(self.device)
        txt_embedding = self.txt_embedding(txts)
        c_t, _ = self.c_t_Model([z, txt_embedding], 
                                      lengths = [input_lengths, txt_lengths + 2])
        output = torch.cat([c_t[i][:, 0, :] for i in range(self.n_modality)], dim=1)
        
        classInput = self.layer_norms_after_concat(output)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)
        output = self.sigmoid(output)

        return output


class LinearClassifier(nn.Module):
    ''' linear classifier '''
    def __init__(self, spk_num):

        super(LinearClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, spk_num)
            #nn.Linear(256, spk_num)
        )

        # def _weights_init(m):
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # self.apply(_weights_init)

    def forward(self, x):
        x = self.classifier(x)

        return F.log_softmax(x, dim=-1)

