import torch.nn as nn
import torch
from torch.autograd import Variable



class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        
        self.input_size = 256
        self.hidden_size = 128
        self.num_layers = 2

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        datasetType = args.train_data_path.split("/")[-2]
        if args.txt_tokenization == "bpe":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(8005, self.input_size)
            elif datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(8005, self.input_size)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(8005, self.input_size)
        elif args.txt_tokenization == "bert":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(30000, self.input_size)
            elif datasetType == "sev_icu":
                raise NotImplementedError
        
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, input_lengths):
        # print("X : ", x.shape)
        embedding = self.linear_embedding(x)
        # print("EM : ", embedding.shape)

        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(embedding)#, (h_0, c_0))

        output_indexes = input_lengths - 1

        outputList = []

        # print("Output : ", output.shape)

        for idx, outMat in enumerate(output):
            outputList.append(outMat[output_indexes[idx]])
        outputList = torch.stack(outputList)

        # print("\nOutput List : ", outputList.shape)

        linOut = self.classifier(outputList)

        # print("Lin Out : ", linOut.shape)

        sigOut = self.sigmoid(linOut)

        # print("Sig Out : ", sigOut.shape)

        return sigOut