import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb


class GRU_D_BINARY(nn.Module):
    def __init__(self, args, bias=True):
        super().__init__()
        
        self.hidden_size = args.hidden_size
        self.output_type = args.output_type
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

        input_size=len(args.vitalsign_labtest)
        self.input_decay = nn.ModuleList()
        for _ in range(input_size):
            self.input_decay.append(nn.Linear(1, 1))

        self.hidden_decay = nn.Linear(input_size, self.hidden_size)

        self.gru = nn.GRUCell(input_size*2, self.hidden_size, bias)

        if self.output_type == "all":
            self.fc_list = nn.ModuleList()
            for _ in range(4):
                self.fc_list.append(nn.Sequential(
                        nn.Linear(in_features=self.hidden_size+2, out_features= self.hidden_size, bias=True),
                        nn.BatchNorm1d(self.hidden_size),
                        self.activations[activation],
                        nn.Linear(in_features=self.hidden_size, out_features= 1,  bias=True)))
        else:
            self.fc = nn.Sequential(
                        nn.Linear(in_features=self.hidden_size+2, out_features= self.hidden_size, bias=True),
                        nn.BatchNorm1d(self.hidden_size),
                        self.activations[activation],
                        nn.Linear(in_features=self.hidden_size, out_features= 1,  bias=True))
        
 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, m, d, x_m, age, gen, length):
        # print("x: ", x.shape)
        # print("h: ", h.shape)
        # print("m: ", m.shape)
        # print("d: ", d.shape)
        # print("x_m: ", x_m.shape)
        # print("age: ", age.shape)
        # print("gen: ", gen.shape)
        # print("length: ", length)

        x_d = []
        for i in range(x.shape[-1]):
            x_d.append(self.input_decay[i](d[:, :, i].unsqueeze(-1)))
        x_d = torch.cat(x_d, dim=-1)
        x_d = torch.exp(-self.relu(x_d))
        x_m = x_m.view(1, 1, -1)

        x = m * x + (1 - m) * x_d * x + (1 - m) * (1 - x_d) * x_m

        output = []
        # print("1: ", x.size(1))
        for i in range(x.size(1)):
            h_d = self.hidden_decay(d[:, i])
            h_d = torch.exp(-self.relu(h_d))
            h = h_d * h
            x_t = torch.cat((x[:, i], m[:, i]), dim=-1)
            h = self.gru(x_t, h)
            output.append(h)

        output = torch.stack(output, dim=1)
        # print("2: ", output.shape)
        output = output[torch.arange(x.shape[0]), length]
        output = torch.cat((output, age.unsqueeze(1), gen.unsqueeze(1)), dim=1)   # gender 추가
        
        if self.output_type == "all":
            multitask_vectors = []
            for i, fc in enumerate(self.fc_list):
                multitask_vectors.append(fc(output))
            output = torch.stack(multitask_vectors)
        else:
            output = self.fc(output)
            
        # print(output)
        # print(output.shape)

        return self.sigmoid(output)
