import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from control.config import args
from itertools import groupby
import torch
from builder.utils.utils import *
import torch.nn.functional as F
from torchvision import transforms
from os.path import exists
from tqdm import tqdm
from bpe import Encoder

dataset = 2

if dataset == 0:
    train_data_path = "/nfs/thena/shared/multi_modal/training_data2/mimic_icu/train"
    test_data_path = "/nfs/thena/shared/multi_modal/training_data2/mimic_icu/test"
elif dataset == 1:
    train_data_path = "/nfs/thena/shared/multi_modal/training_data2/mimic_ed/train"
    test_data_path = "/nfs/thena/shared/multi_modal/training_data2/mimic_ed/test"
elif dataset == 2:
    train_data_path = "/nfs/thena/shared/multi_modal/training_data2/sev_icu/train"
    test_data_path = "/nfs/thena/shared/multi_modal/training_data2/sev_icu/test"

datasetName = train_data_path.split("/")[-2]

fileNm = "vocabList/" + datasetName + "_letters.txt"
outputFile = open(fileNm, "a")

# vocabDict = {}

# brokeChars = {'<', '：', '@', '有', 'ⅱ', '^', '#', '\x9d', '/', '[', ')', ',', '  '}

# index = 0
# with os.scandir(train_data_path) as it:
#     for idx, pkl_path in enumerate(tqdm(it)):
#         if "txt0" in str(pkl_path):
#             continue
    
#         with open(pkl_path, 'rb') as f:
#             data = pkl.load(f)

#         txtInput = data['txt_input']
#         for char in brokeChars:
#             txtInput = txtInput.replace(char, " ")
        
#         words = txtInput.split()
#         for word in words:
#             if word is None:
#                 continue
#             if word in vocabDict:
#                 continue
#             vocabDict[word] = index
#             outputFile.write(word + " " + str(index) + "\n")
#             index += 1

letterDict = {}
brokeChars = {'<', '：', '@', '有', 'ⅱ', '^', '#', '\x9d', '/', '[', ')', ',', '  '}

with os.scandir(train_data_path) as it:
    index = 4
    for idx, pkl_path in enumerate(tqdm(it)):
        if "txt0" in str(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        
        txtInput = data['txt_input']
        
        for char in txtInput:
            if char is None:
                continue
            if char in brokeChars:
                continue
            if char not in letterDict:
                letterDict[char] = index
                outputFile.write(char + " " + str(index) + "\n")
                index += 1

outputFile.close()