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
        if args.model =="retain":
            dataSequence = np.append(reversed(data_pkl['data'][randIndex-randLength+1:randIndex+1]), np.zeros((args.window_size - randLength, 18)), axis=0)
        else:
            dataSequence = np.append(data_pkl['data'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][randIndex-randLength+1:randIndex+1], np.zeros((args.window_size - randLength, 18)), axis=0)
        inputLength = randLength
                
    # Insufficient amount of characters -> more zero-padding at back
    else:
        if args.model =="retain":
            dataSequence = np.append(reversed(data_pkl['data'][:randIndex+1]), np.zeros((windowIndex - randIndex, 18)), axis=0)
        else:
            dataSequence = np.append(data_pkl['data'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        maskSequence = np.append(data_pkl['mask'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        deltaSequence = np.append(data_pkl['delta'][:randIndex+1], np.zeros((windowIndex - randIndex, 18)), axis=0)
        inputLength = randIndex + 1
    
    return dataSequence, maskSequence, deltaSequence, inputLength

def sequenceGenerator_pretrain(args, randIndex, randLength, windowIndex, data_pkl):
    if args.model =="retain":
        print("data_utils.py, Becareful of fusiontype: retain it must be reversed")
        exit(1)
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
    if args.model =="retain":
        print("data_utils.py, Becareful of fusiontype: retain it must be reversed")
        exit(1)
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
