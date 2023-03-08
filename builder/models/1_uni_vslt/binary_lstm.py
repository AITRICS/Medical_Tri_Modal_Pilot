import torch.nn as nn
import torch
from torch.autograd import Variable



class BINARY_LSTM(nn.Module):
    def __init__(self, args):
        super(BINARY_LSTM, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        
        self.input_size = 256
        self.hidden_size = 128
        self.num_layers = 2

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
        
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        self.relu = nn.ReLU()
        # self.classifier = nn.Linear(self.hidden_size, 1)
        self.fc = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features= 64, bias=True),
                    nn.BatchNorm1d(64),
                    self.activations[activation],
                    nn.Linear(in_features=64, out_features= 1,  bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, m, d, x_m, length):
        output, (hn, cn) = self.lstm(x)#, (h_0, c_0))

        # output_indexes = length
        # outputList = []
        # for idx, outMat in enumerate(output):
        #     outputList.append(outMat[output_indexes[idx]])
        # outputList = torch.stack(outputList)
        
        outputList = output[torch.arange(x.shape[0]), length]

        linOut = self.fc(outputList)
        sigOut = self.sigmoid(linOut)
        return sigOut