##문제 존재: collate_fn에서 possible_indexes_key가 빈칸인거 나옴... dataet.py에서 제거했다고 생각했는데..
## ValIndexes.txt 등의 파일 지우고 다시 실행해도..

import pdb
import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import torch.nn.functional as F

from PIL import Image
from os.path import exists
from math import floor, ceil
from itertools import groupby
from monai import transforms
from monai.data import PILReader
# from torchvision import transforms

from builder.utils.utils import *
from control.config import args
from builder.data.collate_fn import *

def collate_binary(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    input_lengths_batch = []
    vslt_len = len(args.vitalsign_labtest)

    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, _, target = data_list
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        pklFeatureMinMaxs[pklFeatureMinMaxs == 0] = 1e-9
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        
        selectedKey = possible_indices_keys[0]
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, args.window_size, windowIndex, data_pkl)
        
        # print("dataSequence: ", dataSequence)
        # print("maskSequence: ", maskSequence)
        # print("initStartIdx: ", data_pkl['initStartIdx'])
        # print("selectedKey: ", selectedKey)
        # print("deltaSequence: ", deltaSequence)
        # print("inputLength: ", inputLength)
        # print(" ")
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                if (i <= selectedKey) and (i > 0):
                    dataSequence[:i, idx] = dataSequence[i, idx]

        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        # final_seqs[0].narrow(0, 0, sample_len).copy_(dataSequence)
        # final_seqs[1].narrow(0, 0, sample_len).copy_(maskSequence)
        # final_seqs[2].narrow(0, 0, sample_len).copy_(deltaSequence)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        seqs_batch.append(final_seqs)    
        target_batch.append(target)
        input_lengths_batch.append(inputLength)

    seqs = torch.stack(seqs_batch)
    targets = torch.Tensor(target_batch).to(torch.long)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, targets, input_lengths

def collate_range(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_sizes, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey])
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-randLength) -1
                
                if (i <= selectedKey) and (i > selectedKey-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                    
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

def collate_range_aux(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_sizes, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey]) 
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex+12, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey+12-randLength) -1
                
                if (i <= selectedKey+12) and (i > selectedKey+12-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                     
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

def collate_range_test(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_size, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = possible_indices_keys[0]
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]
            
        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

def collate_range_txt(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    txt_batch = []
    txt_lengths_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    
    txtDict = txtDictLoad("train")

    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_sizes, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey])
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-randLength) -1
                
                if (i <= selectedKey) and (i > selectedKey-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]

        tokens = txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
        textLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bert":
            if len(tokens) < args.bert_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bert_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bert_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, txts, txt_lengths

def collate_range_txt_aux(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    txt_batch = []
    txt_lengths_batch = []
    f_indices_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    
    txtDict = txtDictLoad("train")

    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_sizes, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey]) 
        dataSequence, maskSequence, deltaSequence, inputLength, f_indices = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex+12, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey+12-randLength) -1
                
                if (i <= selectedKey+12) and (i > selectedKey+12-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
       
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size+12, vslt_len)
        
        # print("seq_data: ", seq_data)
        # df = pd.DataFrame(seq_data)
        # df = df.fillna(method='ffill')
        # data = df.to_numpy()
        # print("data: ", data)

        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))
        
        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]
        # print("multi_target: ", multi_target)
        tokens = txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
        textLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bert":
            if len(tokens) < args.bert_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bert_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bert_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)
        f_indices_batch.append(torch.Tensor(f_indices))

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)
    f_indices_batch = torch.stack(f_indices_batch)

    return seqs, statics, targets, input_lengths, txts, txt_lengths, f_indices_batch

def collate_range_txt_val(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    txt_batch = []
    txt_lengths_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    
    txtDict = txtDictLoad("train")

    for pkl_id, data_list in enumerate(val_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_size, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = possible_indices_keys[0]

        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]
            
        tokens = txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
        textLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bert":
            if len(tokens) < args.bert_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bert_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bert_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, txts, txt_lengths

def collate_range_txt_test(test_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    txt_batch = []
    txt_lengths_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    
    txtDict = txtDictLoad("test")

    for pkl_id, data_list in enumerate(test_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_size, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = possible_indices_keys[0]

        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]
            
        tokens = txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
        textLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bert":
            if len(tokens) < args.bert_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bert_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bert_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, txts, txt_lengths

def collate_seq2seq(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    static_batch = []
    input_lengths_batch = []
    txt_batch = []
    txt_lengths_batch = []

    vslt_len = len(args.vitalsign_labtest)
    txtDict = txtDictLoad("train")

    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, win_sizes = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey]) 
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex+12, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey+12-randLength) -1
                
                if (i <= selectedKey+12) and (i > selectedKey+12-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                    
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size+12, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        tokens = txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
        textLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bert":
            if len(tokens) < args.bert_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bert_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bert_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(final_seqs)
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, input_lengths, txts, txt_lengths

def collate_txt_binary(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    input_lengths_batch = []
    target = 0


    for txtInput in train_data:
        
        split = txtInput.split("/")
        positivities = split[-1].split()
        if args.output_type == "mortality":
            target = int(positivities[0])
        elif args.output_type == "vasso":
            target = int(positivities[2])
        elif args.output_type == "cpr":
            target = int(positivities[4])
        elif args.output_type == "intubation":
            target = int(positivities[6])
        elif args.output_type == "all":
            target = int(int(positivities[0]) or int(positivities[2]) or int(positivities[4]) or int(positivities[6]))
        else:
            raise NotImplementedError
        
        tokens = [int(x) for x in split[1].split()]
        inputLength = len(tokens)
        # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        # EX) 2 {Sentence Tokens} {1 Padding} 3
        # Add Beginnning of Sentence Token
        tokens.insert(0, 2)

        if args.txt_tokenization == "word":
            # Add Padding tokens if too short
            if len(tokens) < args.word_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.word_token_max_length - len(tokens)))
            # Cut if too long
            else:
                tokens = tokens[:args.word_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "character":
            if len(tokens) < args.character_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.character_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.character_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bpe":
            if len(tokens) < args.bpe_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bpe_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bpe_token_max_length-1]
                tokens.append(3)
        elif args.txt_tokenization == "bert":
            if len(tokens) < args.bert_token_max_length - 1:
                tokens.append(3)
                tokens = np.append(tokens, np.ones(args.bert_token_max_length - len(tokens)))
            else:
                tokens = tokens[:args.bert_token_max_length-1]
                tokens.append(3)

        seqs_batch.append(torch.Tensor(tokens))
        target_batch.append(target)
        input_lengths_batch.append(inputLength + 2)

    targets = torch.Tensor(target_batch).to(torch.float)
    seqs = torch.stack(seqs_batch).to(torch.int)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.int)
    
    return seqs, targets, input_lengths

def collate_img_binary(data):
    # initialize batch list
    image_batch  = []
    target_batch = []
    
    # define image preprocess - final shape: (224,224)
    # similar resize & crop process with ImageNet Benchmark 
    for cxr_path, target in data:
        # check image_path: resize & center-crop (224,224)
        img_path = os.path.join(args.image_data_path, 
                                cxr_path.replace('files_jpg','files_resize'))
        if not os.path.exists(img_path):
            print(f'no image: {img_path}')
            continue
        
        # transform (load jpeg img, add channel, rescale 0~1, random rotation)
        preprocess = transforms.Compose([
            transforms.LoadImage(image_only=True, reader=PILReader()),
            transforms.AddChannel(),
            transforms.Transpose((0,2,1)),
            transforms.ScaleIntensity(),
            transforms.RandRotate(range_x=5 * (np.pi / 180), padding_mode='zeros', 
                                    prob=0.5, keep_size=True),   # (-5, 5) degree --> radian
            transforms.ToTensor(),
        ])
        image = preprocess(img_path)
        # save to batch_list 
        image_batch.append(image)
        target_batch.append(target)
    
    # stack images & targets
    images  = torch.stack(image_batch)                  # shape (B, 1, 224, 224)
    targets = torch.Tensor(target_batch).to(torch.long) # shape (B,)
    # debug print
    # print(images.shape, targets.shape)

    return images, targets

def collate_img_binary_test(data):
    # initialize batch list
    image_batch  = []
    target_batch = []
    
    for cxr_path, target in data:
        # check image_path: resize & center-crop (224,224)
        img_path = os.path.join(args.image_data_path, 
                                cxr_path.replace('files_jpg','files_resize'))
        if not os.path.exists(img_path):
            print(f'no image: {img_path}')
            continue
        
        # transform (load jpeg, add channel, rescale 0~1)
        preprocess = transforms.Compose([
            transforms.LoadImage(image_only=True, reader=PILReader()),
            transforms.AddChannel(),
            transforms.Transpose((0,2,1)),
            transforms.ScaleIntensity(),
            transforms.ToTensor(),
        ])
        image = preprocess(img_path)
        # save to batch_list 
        image_batch.append(image)
        target_batch.append(target)

    # stack images & targets
    images  = torch.stack(image_batch)
    targets = torch.Tensor(target_batch).to(torch.long)

    return images, targets

def collate_range_vsltimg(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch = []
    img_lengths_batch=[]
    

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)

    #cxr_li=[]####
    for pkl_id, data_list in enumerate(train_data):
        pkl_path, possible_indices_keys, labels_by_dict, win_size, target = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        randLength = random.randrange(args.min_inputlen, args.window_size+1)
        selectedKey = random.choice(possible_indices_keys)
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        #cxr_li=[]
        #for cxr in data_pkl['cxr_input']:
        #    if cxr[0] <= selectedKey:
        #        cxr_li=data_pkl['cxr_input']####
        #        print(cxr_li)
        #        print('######'*9)
        #        print('cxr_li: ',cxr_li)
        #        time.sleep(0.8)
        #print(data_pkl['cxr_input'])
        #cxr_path=sorted(cxr_li)[-1]
        cxr_path=sorted(data_pkl['cxr_input'])[-1][1]
        #cxr_path = sorted([cxr for cxr in data_pkl['cxr_input'] if cxr[0] <= selectedKey])[-1][1]
        #print('cxr_path: ', cxr_path)
        #print('****'*10)
        img_path = os.path.join(args.image_data_path, 
                                cxr_path.replace('files_jpg','files_resize'))
        #print('img_path: ', img_path)
        if not os.path.exists(img_path):
            print(f'no image: {img_path}')
            continue

        
        # transform (load jpeg img, add channel, rescale 0~1, random rotation)
        preprocess = transforms.Compose([
            transforms.LoadImage(image_only=True, reader=PILReader()),
            transforms.AddChannel(),
            transforms.ScaleIntensity(),
            transforms.RandRotate(range_x=5 * (np.pi / 180), padding_mode='zeros', 
                                    prob=0.5, keep_size=True),   # (-5, 5) degree --> radian
            transforms.ToTensor(),
        ])
        image = preprocess(img_path)

        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-randLength) -1
                
                if (i <= selectedKey) and (i > selectedKey-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        img_batch.append(torch.Tensor(image))

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    imgs = torch.stack(img_batch)

    return seqs, statics, targets, input_lengths, imgs

def collate_range_vsltimg_val(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    

    for pkl_id, data_list in enumerate(val_data):
        pkl_path, possible_indices_keys, labels_by_dict, target, win_size = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = possible_indices_keys[0]

        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        
        cxr_path = sorted([cxr for cxr in data_pkl['cxr_input'] if cxr[0] <= selectedKey])[-1][1]
        img_path = os.path.join(args.image_data_path, 
                                cxr_path.replace('files_jpg','files_resize'))
        if not os.path.exists(img_path):
            print(f'no image: {img_path}')
            continue
        
        # transform (load jpeg img, add channel, rescale 0~1, random rotation)
        preprocess = transforms.Compose([
            transforms.LoadImage(image_only=True, reader=PILReader()),
            transforms.AddChannel(),
            transforms.ScaleIntensity(),
            transforms.ToTensor(),
        ])
        image = preprocess(img_path)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]
            
        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        img_batch.append(torch.Tensor(image))

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    imgs = torch.stack(img_batch)

    return seqs, statics, targets, input_lengths, imgs

def collate_range_vsltimg_test(test_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)
    
    for pkl_id, data_list in enumerate(test_data):
        pkl_path, possible_indices_keys, labels_by_dict, target, win_size = data_list

        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            
        # Static Inputs
        if data_pkl['gender'] == 'M':
            gender = 1
        else:
            gender = 0
        static_inputs = torch.Tensor([gender, data_pkl['age']])

        # Normalization of Data
        pklFeatureMins = args.feature_mins
        pklFeatureMinMaxs = args.feature_maxs - args.feature_mins
        data_pkl['data'] = np.subtract(data_pkl['data'], pklFeatureMins)
        data_pkl['data'] = np.divide(data_pkl['data'], pklFeatureMinMaxs)

        windowIndex = args.window_size - 1
        selectedKey = possible_indices_keys[0]

        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        
        cxr_path = sorted([cxr for cxr in data_pkl['cxr_input'] if cxr[0] <= selectedKey])[-1][1]
        img_path = os.path.join(args.image_data_path, 
                                cxr_path.replace('files_jpg','files_resize'))
        if not os.path.exists(img_path):
            print(f'no image: {img_path}')
            continue
        
        # transform (load jpeg img, add channel, rescale 0~1, random rotation)
        preprocess = transforms.Compose([
            transforms.LoadImage(image_only=True, reader=PILReader()),
            transforms.AddChannel(),
            transforms.ScaleIntensity(),
            transforms.ToTensor(),
        ])
        image = preprocess(img_path)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, args.window_size, vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = neg_multi_target
        else:
            pos_indices = list(set([ceil(oneItv[0] // intv_len) for oneItv in labels_by_dict[selectedKey]]))
            multi_target = [1 if indx in pos_indices else 0 for indx in range(12)]
            
        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)
        img_batch.append(torch.Tensor(image))

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    imgs = torch.stack(img_batch)

    return seqs, statics, targets, input_lengths, imgs

