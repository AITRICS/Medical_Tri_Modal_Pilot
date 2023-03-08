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
#from monai import transforms
#from monai.data import PILReader
from torchvision import transforms

from builder.utils.utils import *
from control.config import args

# randIndex : index of the original data on which the randomly sampled subsequence ends
# randLength : random number for length from which to sample subsequence
# windowIndex : temporary number equal to args.window_size - 1, often used for easy calculation of index values
def sequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl):
    # If there are sufficient characters to sample randLength characters from
    if(randIndex >= randLength - 1):
        dataSequence = np.append(data_pkl['data'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        inputLength = randLength
                
    # Insufficient amount of characters -> more zero-padding at back
    else:
        dataSequence = np.append(data_pkl['data'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        inputLength = randIndex + 1
    
    return dataSequence, maskSequence, deltaSequence, inputLength

def sequenceGenerator_pretrain(args, randIndex, randLength, windowIndex, data_pkl):
    if(randIndex >= randLength - 1):
        dataSequence = np.append(data_pkl['data'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        dataSequence = np.append(dataSequence, data_pkl['data'][randIndex+1:randIndex+13], axis=0)
        
        maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        maskSequence = np.append(maskSequence, data_pkl['mask'][randIndex+1:randIndex+13], axis=0)
        f_indices = np.append((np.sum(maskSequence, 1) > 4), np.zeros(36 - maskSequence.shape[0]), axis=0)
        
        deltaSequence = np.append(data_pkl['delta'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        deltaSequence = np.append(deltaSequence, data_pkl['delta'][randIndex+1:randIndex+13], axis=0)

        length = dataSequence.shape[0]
        if length < 36:
            dataSequence = np.append(dataSequence, np.zeros((36-length, 18)), axis=0)
            maskSequence = np.append(maskSequence, np.zeros((36-length, 18)), axis=0)
            deltaSequence = np.append(deltaSequence, np.zeros((36-length, 18)), axis=0)
        
        inputLength = randLength
                
    # Insufficient amount of characters -> more zero-padding at back
    else:
        dataSequence = np.append(data_pkl['data'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        dataSequence = np.append(dataSequence, data_pkl['data'][randIndex+1:randIndex+13], axis=0)
        
        # maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        maskSequence = np.append(maskSequence, data_pkl['mask'][randIndex+1:randIndex+13], axis=0)
        f_indices = np.append((np.sum(maskSequence, 1) > 4), np.zeros(36 - maskSequence.shape[0]), axis=0)
        
        deltaSequence = np.append(data_pkl['delta'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        deltaSequence = np.append(deltaSequence, data_pkl['delta'][randIndex+1:randIndex+13], axis=0)
        
        length = dataSequence.shape[0]
        if length < 36:
            dataSequence = np.append(dataSequence, np.zeros((36-length, 18)), axis=0)
            maskSequence = np.append(maskSequence, np.zeros((36-length, 18)), axis=0)
            deltaSequence = np.append(deltaSequence, np.zeros((36-length, 18)), axis=0)
        
        inputLength = randIndex + 1
    return dataSequence, maskSequence, deltaSequence, inputLength, f_indices

def testSequenceGenerator(args, randIndex, randLength, windowIndex, data_pkl):
    # If there are sufficient characters to sample randLength characters from
    if(randIndex >= randLength - 1):
        dataSequence = np.append(data_pkl['data'][randIndex-randLength+1:randIndex+1], np.zeros((args.test_window_size - randLength, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.test_window_size - randLength, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][randIndex-randLength+1:randIndex+1], np.zeros((args.test_window_size - randLength, 18)), axis=0)
        inputLength = randLength
                
    # Insufficient amount of characters -> more zero-padding at back
    else:
        dataSequence = np.append(data_pkl['data'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        inputLength = randIndex + 1
    
    return dataSequence, maskSequence, deltaSequence, inputLength

def txtDictLoad(flowType):
    datasetName = args.train_data_path.split("/")[-2]
    tokenType = args.txt_tokenization

    txtDatasetName = "builder/data/text/textDataset/mimic_icu_" + flowType + "_" + tokenType + "_" + "textDataset.txt"
    txtDict = {}

    txtFile = open(txtDatasetName, "r")
    k = 0
    while True:
        line = txtFile.readline()
        if not line:
            break

        line = line.strip().split("/")
        patId = int(line[0].split()[0])
        chid = int(line[0].split()[1])

        tokens = [int(x) for x in line[1].split()]
        txtDict[(patId, chid)] = tokens

        k += 1
    txtFile.close()
    return txtDict

##### unimodal vslt collate fn #####      
def collate_within_vslt(train_data):
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
        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        else:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex, data_pkl)
        
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

def collate_within_vslt_test(train_data):
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)

        seqs_batch.append(final_seqs)
        target_batch.append(torch.tensor(multi_target))
        input_lengths_batch.append(inputLength)
        static_batch.append(static_inputs)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths

##### bimodal vslt+txt collate fn #####      
def collate_within_vslttxt(train_data):
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
        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        else:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex, data_pkl)
        
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

def collate_within_vslttxt_val(val_data):
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

        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        else:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex, data_pkl)
        
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

def collate_within_vslttxt_test(test_data):
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

        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        else:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex, data_pkl)
        
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

##### bimodal vslt+img collate fn #####
def collate_within_vsltimg(train_data, transform):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch=[]

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

        if 'train-full' in args.modality_inclusion:
            earliest_img_time = min([j[0] for j in data_pkl['cxr_input']])
            possible_indices_keys= list([i for i in possible_indices_keys if earliest_img_time<=i])
    
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey])
        # dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        else:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex, data_pkl)

        if 'train-full' in args.modality_inclusion:
            cxr_li = [i for i in data_pkl['cxr_input'] if i[0] <= selectedKey] 
            if not cxr_li: 
                print("collate cxr error")
                exit(1)
            cxr_path = sorted(cxr_li)[-1][1]
        else:
            cxr_path = None
        
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)###
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)###

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
    # print(imgs.size())#256 x 224확인'

    return seqs, statics, targets, input_lengths, imgs

def collate_within_vsltimg_val(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch=[]

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)

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
        if 'train-full' in args.modality_inclusion:
            earliest_img_time = min([j[0] for j in data_pkl['cxr_input']])
            if earliest_img_time <= selectedKey:
                pass
            else:
                print("collate cxr error")
                exit(1)

        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        # cxr_li = [data_pkl['cxr_input'][i] for i in range(len(data_pkl['cxr_input'])) if data_pkl['cxr_input'][i][0] <= selectedKey] 
        # if not cxr_li:
        #     continue
        cxr_path = sorted(cxr_li)[-1][1]
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)###
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)###

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

def collate_within_vsltimg_test(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch=[]

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)

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
        if 'test-full' in args.modality_inclusion:
            earliest_img_time = min([j[0] for j in data_pkl['cxr_input']])
            if earliest_img_time <= selectedKey:
                pass
            else:
                print("collate cxr error")
                exit(1)

        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        # cxr_li = [data_pkl['cxr_input'][i] for i in range(len(data_pkl['cxr_input'])) if data_pkl['cxr_input'][i][0] <= selectedKey] 
        # if not cxr_li:
        #     continue
        cxr_path = sorted(cxr_li)[-1][1]
        img_path = os.path.join(args.image_data_path, 
                                cxr_path.replace('files_jpg','files_margins'))
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)###
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)###

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

##### Trimodal collate fn #####
def collate_within_vsltimgtxt(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch=[]
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
        # randLength = random.randrange(args.min_inputlen, args.window_size+1)
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey])
        # dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
        else:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex, data_pkl)
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)###
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)###
        
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
        img_batch.append(torch.Tensor(image))
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    imgs = torch.stack(img_batch)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, imgs, txts, txt_lengths

def collate_within_vsltimgtxt_val(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch=[]
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
        # randLength = random.randrange(args.min_inputlen, args.window_size+1)
        selectedKey = possible_indices_keys[0]
        
        #randLength = random.choice(win_sizes[selectedKey])
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)###
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)###

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
        img_batch.append(torch.Tensor(image))
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    imgs = torch.stack(img_batch)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, imgs, txts, txt_lengths

def collate_within_vsltimgtxt_test(val_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    seqs_batch = []
    target_batch = []
    static_batch = []
    input_lengths_batch = []
    img_batch=[]
    txt_batch = []
    txt_lengths_batch = []

    vslt_len = len(args.vitalsign_labtest)
    neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intv_len = int(args.prediction_range // 12)

    txtDict = txtDictLoad("test")

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
        # randLength = random.randrange(args.min_inputlen, args.window_size+1)
        selectedKey = possible_indices_keys[0]
        
        #randLength = random.choice(win_sizes[selectedKey])
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
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // intv_len)###
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)###

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
        img_batch.append(torch.Tensor(image))
        txt_batch.append(torch.Tensor(tokens))
        txt_lengths_batch.append(textLength)

    seqs = torch.stack(seqs_batch)
    statics = torch.stack(static_batch)
    targets = torch.stack(target_batch)
    input_lengths = torch.Tensor(input_lengths_batch).to(torch.long)
    imgs = torch.stack(img_batch)
    txts = torch.stack(txt_batch)
    txt_lengths = torch.Tensor(txt_lengths_batch).to(torch.long)

    return seqs, statics, targets, input_lengths, imgs, txts, txt_lengths

