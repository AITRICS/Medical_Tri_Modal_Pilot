import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb


class GRU_D_ATTN_V1(nn.Module):
    def __init__(self, args, input_size=17, hidden_size=64, bias=True):
        super().__init__()

        self.input_decay = nn.ModuleList()
        for _ in range(input_size):
            self.input_decay.append(nn.Linear(1, 1))

        # Initial Fully Connected Layers
        self.init_fc_list = nn.ModuleList()
        for _ in range(input_size):
            self.init_fc_list.append(nn.Linear(args.window_size, 128))
        self.agvpool = nn.AdaptiveAvgPool2d((1,input_size))
        self.attn_values = 0

        self.hidden_decay = nn.Linear(input_size, hidden_size)

        self.gru = nn.GRUCell(input_size*2, hidden_size, bias)

        self.fc = nn.Linear(hidden_size+2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, m, d, x_m, age, gender, length):
        x_d = []
        for i in range(x.shape[-1]):
            x_d.append(self.input_decay[i](d[:, :, i].unsqueeze(-1)))
        x_d = torch.cat(x_d, dim=-1)
        x_d = torch.exp(-self.relu(x_d))
        x_m = x_m.view(1, 1, -1)
        
        x = m * x + (1 - m) * x_d * x + (1 - m) * (1 - x_d) * x_m

        attn_map = []
        for i, init_fc in enumerate(self.init_fc_list):
            attn_map.append(init_fc(x[:, :, i]))
        attn_map = torch.stack(attn_map, 2)
        attn_map = self.agvpool(attn_map).squeeze(1)
        attn_map = F.softmax(attn_map, dim=1)
        self.attn_values = attn_map
        x = x * attn_map.unsqueeze(1)

        output = []
        for i in range(x.size(1)):
            h_d = self.hidden_decay(d[:, i])
            h_d = torch.exp(-self.relu(h_d))
            h = h_d * h
            x_t = torch.cat((x[:, i], m[:, i]), dim=-1)
            h = self.gru(x_t, h)
            output.append(h)

        output = torch.stack(output, dim=1)
        output = output[torch.arange(x.shape[0]), length]
        output = torch.cat((output, age.unsqueeze(1), gender.unsqueeze(1)), dim=1)   # gender 추가
        output = self.fc(output)
        exit(1)

        return self.sigmoid(output)
