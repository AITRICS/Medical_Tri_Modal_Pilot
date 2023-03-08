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
        nn.Conv2d(in_planes, planes, kernel_size=(1,9), stride=(1,stride), padding=(0,4), bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(),
        nn.Conv2d(planes, planes, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
        nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                        kernel_size=1, stride=(1,stride), bias=False),
                nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = self.net(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN2D_RESNET_V1(nn.Module):
        def __init__(self, args, device):
                super(CNN2D_RESNET_V1, self).__init__()      
                self.args = args
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

                def conv2d_bn(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm2d(oup),
                                self.activations[activation],
                                nn.Dropout(self.dropout),
                )
                self.in_planes = 64
                self.features = nn.Sequential(
                        conv2d_bn(1,  64, (1,51), (1,1), (0,25)), 
                        nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                )
                block = BasicBlock
                self.layer1 = self._make_layer(block, 64, 2, stride=1)
                self.layer2 = self._make_layer(block, 128, 2, stride=2)
                self.layer3 = self._make_layer(block, 256, 2, stride=2)
                self.layer4 = self._make_layer(block, 512, 2, stride=2)

                self.agvpool = nn.AdaptiveAvgPool2d((8,1))

                self.classifier = nn.Sequential(
                        nn.Linear(in_features=4096, out_features= 256, bias=True),
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
                x = x.reshape(x.size(0), -1)
                output = self.classifier(x)
                return output
                
                
        def init_state(self, device):
                pass
         