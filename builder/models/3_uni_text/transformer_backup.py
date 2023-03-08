import torch.nn as nn
import torch
from builder.models.src.transformer import *
import pickle
import matplotlib.pyplot as plt

from builder.models.src.transformer.module import PositionalEncoding
from bpe import Encoder

class TransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head,
                 d_model, d_ff, dropout=0.1, pe_maxlen=5000, use_pe=True, block_mask=None):
        super(TransformerEncoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen
        self.use_pe = use_pe
        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=n_head,
                d_ff=d_ff,
                dropout_p=dropout,
                block_mask=block_mask,
            ) for _ in range(n_layers)
        ])

    def forward(self, padded_input, input_lengths=None, return_attns=True):
        enc_slf_attn_list = []

        #Prepare masks
        # non_pad_mask = get_non_pad_mask(padded_input, 
        #                                 input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        #print(non_pad_mask)
        #print(slf_attn_mask)

        # Forward
        if self.use_pe:
            enc_output = self.dropout(padded_input + self.positional_encoding(padded_input.size(1)))
            # enc_output = self.dropout(
            #     self.layer_norm_in(self.linear_in(padded_input)) +
            #     self.positional_encoding(padded_input.size(1)))
        else:
            enc_output = self.dropout(
                self.layer_norm_in(self.linear_in(padded_input)))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output


class T_TRANSFORMER_V1(nn.Module):
    def __init__(self, args):
        super(T_TRANSFORMER_V1, self).__init__()      
        self.args = args

        self.num_layers = args.txt_num_layers
        self.dropout = args.txt_dropout
        self.model_d = args.txt_model_dim
        self.num_heads = args.txt_num_heads
        self.classifier_nodes = args.txt_classifier_nodes

        self.transformer_encoder = TransformerEncoder(
                                        d_input=self.model_d,
                                        n_layers=self.num_layers,
                                        n_head=self.num_heads,
                                        d_model=self.model_d,
                                        d_ff=self.model_d*2,
                                        dropout=self.dropout,
                                        pe_maxlen=500)


        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.model_d, out_features=self.classifier_nodes, bias=True),
            nn.BatchNorm1d(self.classifier_nodes),
            nn.ReLU(),
            nn.Linear(in_features=self.classifier_nodes, out_features=1, bias=True)
        )

        # if args.txt_tokenization == "word":
        #     self.classifier = nn.Sequential(
        #         nn.Linear(in_features=self.model_d * args.word_token_max_length, out_features=self.classifier_nodes, bias=True),
        #         nn.BatchNorm1d(self.classifier_nodes),
        #         nn.ReLU(),
        #         nn.Linear(in_features=self.classifier_nodes, out_features=2, bias=True),
        #     )
        # elif args.txt_tokenization == "character":
        #     self.classifier = nn.Sequential(
        #         nn.Linear(in_features=self.model_d * args.character_token_max_length, out_features=self.classifier_nodes, bias=True),
        #         nn.BatchNorm1d(self.classifier_nodes),
        #         nn.ReLU(),
        #         nn.Linear(in_features=self.classifier_nodes, out_features=2, bias=True)    
        #     )
        # elif args.txt_tokenization == "bpe":
        #     self.classifier = nn.Sequential(
        #         nn.Linear(in_features=self.model_d * args.bpe_token_max_length, out_features=self.classifier_nodes, bias=True),
        #         nn.BatchNorm1d(self.classifier_nodes),
        #         nn.ReLU(),
        #         nn.Linear(in_features=self.classifier_nodes, out_features=2, bias=True)
        #     )

        datasetType = args.train_data_path.split("/")[-2]
        if args.txt_tokenization == "word":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(1620, self.model_d)
            elif datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(3720, self.model_d)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(45282, self.mode_d)
        elif args.txt_tokenization == "character":
            if datasetType == "mimic_icu" or datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(42, self.model_d)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(1130, self.model_d)
        elif args.txt_tokenization == "bpe":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(8005, self.model_d)
            elif datasetType == "mimic_ed":
                self.linear_embedding = nn.Embedding(8005, self.model_d)
            elif datasetType == "sev_icu":
                self.linear_embedding = nn.Embedding(8005, self.model_d)
        elif args.txt_tokenization == "bert":
            if datasetType == "mimic_icu":
                self.linear_embedding = nn.Embedding(30000, self.model_d)
            elif datasetType == "sev_icu":
                raise NotImplementedError
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.model_d)).to(self.args.device)
        # self.cls_tokens = self.cls_token
        # for _ in range(self.args.batch_size - 1):
        #     self.cls_tokens = torch.cat((self.cls_tokens, self.cls_token), 0)

        self.tmp_counter_int = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, input_lengths):
        self.tmp_counter_int += 1

        #print(x.shape)
        embedding = self.linear_embedding(x)
        #embedding = embedding.squeeze()
        # print("1: ", embedding.shape)
        # print("1: ", embedding[0])

        cls_tokens = self.cls_tokens.repeat(self.args.batch_size, 1, 1)

        # print("Embedding", embedding.shape)
        # print("CLS Tokens", cls_tokens.shape)

        embeddingWithCls = torch.cat([cls_tokens, embedding], 1)
        # print("2: ", embeddingWithCls.shape)

        enc_output, attention_map = self.transformer_encoder(embeddingWithCls, input_lengths=input_lengths+1)
        #enc_output = enc_output.squeeze()
        #print(enc_output.shape)

        # if self.tmp_counter_int < 5:
        #     for attnNum, attention in enumerate(attention_map):
        #         print(attnNum, attention.shape)

        if self.args.txt_tokenization == "bpeasfddsdasad":

            if self.tmp_counter_int % 1000 == 1:
                encoderFile = open("builder/data/text/textDatasetEncoder/mimic_icu_8000.obj", 'rb')
                encoderObj = pickle.load(encoderFile)

                for senNum, sentence in enumerate(x):
                    wordArr = ["CLS", "SOS"]
                    tmpArr = []
                    for tok in sentence:
                        if tok >= 8000:
                            tmpArr.append(tok - 8000)
                        elif tok == 3:
                            break
                        elif tok == 2:
                            continue
                        else:
                            tmpArr.append(tok)
                    tmpArr = [[dfg.item() for dfg in tmpArr]]
                    # print(tmpArr)
                    t1 = next(encoderObj.inverse_transform(tmpArr))
                    #print(t1)
                    t2 = encoderObj.tokenize(t1)
                    #print(t2)
                    wordArr += t2
                    wordArr.append("EOS")
                    for tmpI in range(10):
                        wordArr.append("PAD")
                    wordArr = wordArr[:10]
                    # print(wordArr)

                    attentionL1M1 = attention_map[0][0]
                    attentionL1M1 = [pls[:input_lengths[senNum]] for pls in attentionL1M1[:input_lengths[senNum]]]
                    attentionL1M2 = attention_map[0][1]

                    # plt.imshow(attentionL1M1, cmap="hot", interpolation="nearest")
                    # plt.savefig('t.png')

                    # print(attentionL1M1)
                    # if senNum == 0:
                    #     print(attentionL1M1)
                    #     exit(1)


        #flattened_enc_output = torch.flatten(enc_output, start_dim=1, end_dim=-1)
        #print(flattened_enc_output.shape)

        clsOutput = enc_output[:,0]
        # print(clsOutput.shape)
        # print(clsOutput)

        output = self.classifier(clsOutput)
        #output = output.squeeze()
        #print(type(output))
        #print(nn.functional.softmax(output, dim=1))

        #return nn.functional.softmax(output, dim=1)

        # print(output.shape)
        sigOut = self.sigmoid(output)
        return sigOut