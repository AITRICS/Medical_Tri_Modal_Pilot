import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb


class BINARY_GRU_D(nn.Module):
    def __init__(self, args, bias=True):
        super().__init__()
        
        self.hidden_size = args.hidden_size
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

        self.fc = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features= 64, bias=True),
                    nn.BatchNorm1d(64),
                    self.activations[activation],
                    nn.Linear(in_features=64, out_features= 1,  bias=True))
        
        self.relu = self.activations[activation]
        self.sigmoid = self.activations['sigmoid']

    def forward(self, x, h, m, d, x_m, length):
        # print("x: ", x.is_cuda)
        # print("h: ", h.is_cuda)
        # print("m: ", m.is_cuda)
        # print("d: ", d.is_cuda)
        # print("x_m: ", x_m.is_cuda)
        # print("length: ", length.is_cuda)

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
        output = output[torch.arange(x.shape[0]), length]
        output = self.fc(output)

        return self.sigmoid(output)
