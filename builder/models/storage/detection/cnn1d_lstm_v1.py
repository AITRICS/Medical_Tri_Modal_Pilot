import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.net = nn.Sequential(
        nn.Conv1d(in_planes, planes, kernel_size=9, stride=stride, padding=4, bias=False),
        nn.BatchNorm1d(planes),
        nn.ReLU(),
        nn.Conv1d(planes, planes, kernel_size=9, stride=1, padding=4, bias=False),
        nn.BatchNorm1d(planes)
        )

        self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        out = self.net(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN1D_LSTM_V1(nn.Module):
        def __init__(self, args, device):
                super(CNN1D_LSTM_V1, self).__init__()      
                self.args = args

                self.num_layers = 2
                self.hidden_dim = 512
                self.dropout = 0.1
                self.feature_extractor = args.enc_model

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

                # Create a new variable for the hidden state, necessary to calculate the gradients
                self.hidden = ((torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)))

                def conv1d_bn(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv1d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm1d(oup),
                                self.activations[activation],
                                nn.Dropout(self.dropout),
                )
                self.in_planes = 64
                self.features = nn.Sequential(
                        conv1d_bn(1,  64, 51, 1, 25), 
                        nn.MaxPool1d(kernel_size=4, stride=4),
                        # conv1d_bn(64, 128, 21, 1, 10),
                        # conv1d_bn(128, 256, 21, 2, 10),
                        # conv1d_bn(256, 512, 21, 1, 10),
                )
                block = BasicBlock
                self.layer1 = self._make_layer(block, 64, 2, stride=1)
                self.layer2 = self._make_layer(block, 128, 2, stride=2)
                self.layer3 = self._make_layer(block, 256, 2, stride=2)
                self.layer4 = self._make_layer(block, 512, 2, stride=2)


                self.agvpool = nn.AdaptiveAvgPool1d(10)

                self.lstm = nn.LSTM(
                        input_size=512,
                        hidden_size=self.hidden_dim,
                        num_layers=self.num_layers,
                        batch_first=True,
                        dropout=self.dropout) 

                self.classifier = nn.Sequential(
                        nn.Linear(in_features=self.hidden_dim, out_features= 256, bias=True),
                        nn.BatchNorm1d(256),
                        self.activations[activation],
                        nn.Linear(in_features=256, out_features= args.output_dim, bias=True),
                )
        def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride1 in strides:
                        layers.append(block(self.in_planes, planes, stride1))
                        self.in_planes = planes
                return nn.Sequential(*layers)
        
        def forward(self, x):
                x = x.unsqueeze(1)
                x = self.features(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.agvpool(x)
                x = torch.squeeze(x, 2)
                x = x.permute(0, 2, 1)
                self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
                output, self.hidden = self.lstm(x, self.hidden)    
                output = output[:,-1,:]
                output = self.classifier(output)
                return output
                
                
        def init_state(self, device):
                self.hidden = ((torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)))
         