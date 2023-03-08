import torch.nn as nn
import torch
from torch.autograd import Variable



class MULTI_LSTM(nn.Module):
    def __init__(self, args):
        super(MULTI_LSTM, self).__init__()
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
        self.fc_list = nn.ModuleList()
        for _ in range(12):
            self.fc_list.append(nn.Sequential(
                nn.Linear(in_features=self.hidden_size+2, out_features= self.hidden_size, bias=True),
                nn.BatchNorm1d(self.hidden_size),
                self.activations[activation],
                nn.Linear(in_features=self.hidden_size, out_features= 1,  bias=True)))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, m, d, x_m, age, gen, length):
        output, (hn, cn) = self.lstm(x)#, (h_0, c_0))

        # output_indexes = length
        # outputList = []
        # for idx, outMat in enumerate(output):
        #     outputList.append(outMat[output_indexes[idx]])
        # outputList = torch.stack(outputList)

        outputList = output[torch.arange(x.shape[0]), length]
        output = torch.cat((outputList, age.unsqueeze(1), gen.unsqueeze(1)), dim=1)

        multitask_vectors = []
        for i, fc in enumerate(self.fc_list):
            multitask_vectors.append(fc(output))
        output = torch.stack(multitask_vectors)
        
        # return self.sigmoid(output)
        return output

