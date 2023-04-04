import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RETAIN(nn.Module):
    	# def __init__(self, dim_input, dim_emb=128, dropout_input=0.8, dropout_emb=0.5, dim_alpha=128, dim_beta=128,
		# 		 dropout_context=0.5, dim_output=2, l2=0.0001, batch_first=True):
    def __init__(self, args, batch_first=True, dropout_input=0.8, dropout_emb=0.5, dim_alpha=256, dim_beta=256,
                    dropout_context=0.5, dim_output=1, l2=0.0001): #dim_output 2를 1로 바꿈 0과 1 중에 고르는 것
        super().__init__()
        # super(RETAIN, self).__init__()

        self.args = args
        self.hidden_size = args.hidden_size
        input_size=len(self.args.vitalsign_labtest)
        self.batch_first = batch_first

        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_input),
            nn.Linear(input_size, self.hidden_size, bias=False),
            nn.Dropout(p=dropout_emb)
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=self.hidden_size, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.GRU(input_size=self.hidden_size, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=self.hidden_size)
        init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=self.hidden_size + 2, out_features=dim_output)
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()
        
        # activation = 'relu'
        # self.activations = nn.ModuleDict([
        #         ['lrelu', nn.LeakyReLU()],
        #         ['prelu', nn.PReLU()],
        #         ['relu', nn.ReLU(inplace=True)],
        #         ['tanh', nn.Tanh()],
        #         ['sigmoid', nn.Sigmoid()],
        #         ['leaky_relu', nn.LeakyReLU(0.2)],
        #         ['elu', nn.ELU()]
        # ])
        # self.fc = nn.Sequential(
        #         nn.Linear(in_features=self.hidden_size+2, out_features= self.hidden_size, bias=True),
        #         nn.BatchNorm1d(self.hidden_size),
        #         self.activations[activation],
        #         nn.Linear(in_features=self.hidden_size, out_features= 1,  bias=True))
    def forward(self, x, h, m, d, x_m, age, gen, length, x_txt, txt_lengths, x_img, exist_img, missing_num, feasible_indices, img_time, txt_time, flow_type, reports_tokens, reports_lengths):#self, x, lengths
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

		# emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)
        
        packed_input = pack_padded_sequence(emb, length.to("cpu"), batch_first=self.batch_first, enforce_sorted=False).to(self.args.device)
        g, _ = self.rnn_alpha(packed_input)

		# alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first, total_length=max_len)

		# mask -> batch_size X max_len X 1
        mask = Variable(torch.FloatTensor(
			[[1.0 if i < length[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
						requires_grad=False)
        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()

		# e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

		# Alpha = batch_size X max_len X 1
		# alpha value for padded visits (zero) will be zero
        alpha = masked_softmax(e, mask)
        
        h, _ = self.rnn_beta(packed_input)

		# beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first, total_length=max_len)

		# Beta -> batch_size X max_len X dim_emb
		# beta for padded visits will be zero-vectors
        beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

		# context -> batch_size X (1) X dim_emb (squeezed)
		# Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
		# Vectorized sum
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

		# without applying non-linearity
        context = torch.cat((context, age.unsqueeze(1), gen.unsqueeze(1)), dim=1)
        logit = self.output(context)

        return logit, None