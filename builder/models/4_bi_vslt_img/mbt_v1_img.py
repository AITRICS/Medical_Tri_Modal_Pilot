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
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from control.config import args
# late concat V-trans T-trans

class MBT_V1_IMG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size = args.image_size
        img_hidden_size = 256
        patch_size = 16
        img_num_heads = 4
        pos_embed = "conv"

        self.n_modality = 2

        self.num_layers = 12 #earlyfusion에서는 transformer_num_layers
        self.num_heads = args.transformer_num_head
        self.model_dim = args.transformer_dim
        self.dropout = args.dropout

        self.num_nodes = len(args.vitalsign_labtest)
        self.t_len = args.window_size

        self.device = args.device
        self.classifier_nodes = 64

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

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=img_hidden_size,
            num_heads=img_num_heads,
            pos_embed=pos_embed,
            dropout_rate=0,
            spatial_dims=2,
        )

        self.vslt_encoder = MBTEncoder(###
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
        
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
            nn.Linear(in_features=self.model_dim * 2, out_features= 64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features= 1,  bias=True)))

        self.relu = self.activations[activation]
        # self.sigmoid = self.activations['sigmoid']

        
        
        #self.linear_embedding = nn.Linear(self.num_nodes+2, self.model_dim)
        self.init_fc = nn.Sequential(
                                    nn.Linear(self.num_nodes+2, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.model_dim, self.model_dim),
                                    nn.LayerNorm(self.model_dim),
                                    nn.ReLU(inplace=True),
                )


    def forward(self, x, h, m, d, x_m, age, gen, input_lengths, img):#####
        from matplotlib import pyplot 
        import matplotlib.pyplot as plt
        
        a = img[0].permute(1,2,0)
        #print(a)
        print(a.size())
        print(a.detach().cpu())
        
        plt.imshow(a.detach().cpu(), cmap='gray')
        plt.show()

        #plt.imshow(a.detach().cpu(), cmap=pyplot.cm.binary)
        #plt.show()
        
        age = age.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        gen = gen.unsqueeze(1).unsqueeze(2).repeat(1, x.size(1), 1)
        x = torch.cat([x, age, gen], axis=2)
        #print(x.size())
        #print(x)
        vslt_embedding = self.init_fc(x)
        #vslt_embedding = self.linear_embedding(x)
        img_embedding = self.patch_embedding(img)

        #final_input = torch.cat([vslt_embedding, img_embedding], 1)
        outputs, _=self.vslt_encoder([vslt_embedding, img_embedding],lengths=[input_lengths, torch.tensor([img_embedding.size(1)]*img_embedding.size(0))+2])
        final_output = torch.cat([outputs[i][:, 0, :] for i in range(self.n_modality)], dim=1)       

        #txts = txts.type(torch.IntTensor).to(self.device)####
        #txt_embedding = self.txt_embedding(txts)####
        

        
        #ageTensor = age.unsqueeze(1)
        #genTensor = gen.unsqueeze(1)
        #final_output = torch.cat((final_output, ageTensor, genTensor), 1)
        classInput = self.layer_norms_after_concat(final_output)### 여기까지

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(classInput))
        output = torch.stack(multitask_vectors)
        # output = self.sigmoid(output)
        #exit(1)
        return output