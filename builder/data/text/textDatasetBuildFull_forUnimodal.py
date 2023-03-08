import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from itertools import groupby
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from os.path import exists
from tqdm import tqdm
from math import floor
from bpe import Encoder
from lib2to3.pgen2.tokenize import tokenize
from transformers import AutoModel, AutoTokenizer

# Parameters for Running File
dataset = 0
dtype = "train"
token_type = "bpe"

k = {}

def DatetimeToHours(time):
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    result = (year * 8760) + (month * 730) + (day * 24) + hour + (minute / float(60))
    return result

if dataset == 0:
    k["train_data_path"] = "/nfs/thena/shared/multi_modal/training_data_0725/mimic_icu/train"
    k["test_data_path"] = "/nfs/thena/shared/multi_modal/training_data_0725/mimic_icu/test"
elif dataset == 1:
    k["train_data_path"] = "/nfs/thena/shared/multi_modal/training_data_0706/mimic_ed/train"
    k["test_data_path"] = "/nfs/thena/shared/multi_modal/training_data_0706/mimic_ed/test"
elif dataset == 2:
    k["train_data_path"] = "/nfs/thena/shared/multi_modal/training_data_0725/sev_icu/train"
    k["test_data_path"] = "/nfs/thena/shared/multi_modal/training_data_0725/sev_icu/test"

datasetName = k[dtype + "_data_path"].split("/")[-2]
data_path = k[dtype + "_data_path"]
test_data_path = k["test_data_path"]

fileNm = "builder/data/text/textDataset/" + datasetName + "_" + dtype +"_" + token_type +"_full_textDataset.txt"
outputFile = open(fileNm, "w")

# {pat_id} {chid} /{token numbers separeted by spaces}/ {death_yn} {death_time} {vasso_yn} {vasso_time} {cpr_yn} {cpr_time} {intubation_yn} {intubation time}
# -> Time is set to -1 if target is not positive
# EX) 1239812 398493 /3 4523 231 54 1 3247 99/ 0 -1 1 23 1 349 0 -1
# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

if token_type == "word":
    vocabFileNm = "builder/data/text/vocabList/" + datasetName + "_vocab.txt"
    vocabFile = open(vocabFileNm, "r")

    vocabDict = {}
    while True:
        line = vocabFile.readline()
        if not line:
            break
        line = line.strip().split()
        word = line[0]
        key = int(line[1])
        vocabDict[word] = key

    vocabFile.close()

    brokeChars = {'<', '：', '@', '有', 'ⅱ', '^', '#', '\x9d', '/', '[', ')', ',', '  '}

    index = 0
    with os.scandir(data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it)):
            if "txt0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            
            txtInput = data['txt_input']
            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")

            printStr = data['pat_id'] + " " + data['chid'] + " /"

            words = txtInput.split()
            for word in words:
                if word in vocabDict:
                    printStr += str(vocabDict[word]) + " "
                else:
                    printStr += "0 "
            
            if data['death_yn'] == 1:
                #print(data['death_time'], type(data['death_time']))
                death_time = int(floor(float(data['death_time'])))
            else:
                death_time = None

            for i, x in enumerate(data['vasso_inputs']):
                if x == 1:
                    vasso_time = i
                    vassoTime_yn = 1
                    break
                if i == len(data['vasso_inputs']) - 1:
                    vassoTime_yn = 0
            for i, x in enumerate(data['cpr_inputs']):
                if x == 1:
                    cpr_time = i
                    cprTime_yn = 1
                    break
                if i == len(data['cpr_inputs']) - 1:
                    cprTime_yn = 0
            for i, x in enumerate(data['intubation_inputs']):
                if x == 1:
                    intubation_time = i
                    intubationTime_yn = 1
                    break
                if i == len(data['intubation_inputs']) - 1:
                    intubationTime_yn = 0

            printStr = printStr.strip()
            printStr += "/ "
            printStr += str(data['death_yn'])
            if death_time is None:
                printStr += " -1"
            else:
                printStr += " " + str(death_time)
            printStr += " " + str(data['vasso_yn'])
            if vassoTime_yn == 1:
                printStr += " " + str(vasso_time)
            else:
                printStr += " -1"
            printStr += " " + str(data['cpr_yn'])
            if cprTime_yn == 1:
                printStr += " " + str(cpr_time)
            else:
                printStr += " -1"
            printStr += " " + str(data['intubation_yn'])
            if intubationTime_yn == 1:
                printStr += " " + str(intubation_time)
            else:
                printStr += " -1"
            printStr += "\n"

            outputFile.write(printStr)

if token_type == "character":
    vocabFileNm = "builder/data/text/vocabList/" + datasetName + "_letters.txt"
    vocabFile = open(vocabFileNm, "r")

    trainMor = [0,0]
    trainVas = [0,0]
    trainCpr = [0,0]
    trainInt = [0,0]
    testMor = [0,0]
    testVas = [0,0]
    testCpr = [0,0]
    testInt = [0,0]

    vocabDict = {}
    while True:
        line = vocabFile.readline()
        if not line:
            break
        line = line.rstrip()
        char = line[0]
        key = int(line.split()[-1])
        vocabDict[char] = key
    vocabFile.close()

    brokeChars = {'<', '：', '@', '有', 'ⅱ', '^', '#', '\x9d', '/', '[', ')', ',', '  '}

    with os.scandir(data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it)):
            if "txt0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)

            TimeBetweenAdmVital = data['window_first_idx_hr'] - DatetimeToHours(data['admission_time'])
            if TimeBetweenAdmVital < 0 or TimeBetweenAdmVital > 21:
                continue

            endTime = 36 - TimeBetweenAdmVital
            startTime = -TimeBetweenAdmVital

            txtInput = data['txt_input'][1]
            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")
            
            printStr = data['pat_id'] + " " + data['chid'] + " /"

            for char in txtInput:
                if char in vocabDict:
                    printStr += str(vocabDict[char]) + " "
                else:
                    printStr += "0 "
            
            printStr = printStr.strip()
            printStr += "/ "

            if data['death_yn'] == 1 and data['death_time'] is None:
                data['death_yn'] = 0
            elif data['death_yn'] == 1 and data['death_time'] > 96 - TimeBetweenAdmVital:
                data['death_yn'] = 0

            if data['death_yn'] == 1:
                #print(data['death_time'], type(data['death_time']))
                death_time = int(floor(float(data['death_time'])))
            else:
                death_time = None
            
            if data['vasso_yn'] == 1 and data['vasso_time'] is None:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1 and len(data['vasso_time']) < 1:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1:
                withinTime = False
                time = None
                for t in data['vasso_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['vasso_yn'] = 0

            if data['cpr_yn'] == 1 and data['cpr_time'] is None:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1 and len(data['cpr_time']) < 1:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1:
                withinTime = False
                time = None
                for t in data['cpr_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['cpr_yn'] = 0

            if data['intubation_yn'] == 1 and data['intubation_time'] is None:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1 and len(data['intubation_time']) < 1:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1:
                withinTime = False
                time = None
                for t in data['intubation_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['intubation_yn'] = 0
            
            printStr += str(data['death_yn'])
            if death_time is None:
                printStr += " -1"
            else:
                printStr += " " + str(death_time)

            printStr += " " + str(data['vasso_yn'])
            if data['vasso_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['cpr_yn'])
            if data['cpr_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['intubation_yn'])
            if data['intubation_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += "\n"

            outputFile.write(printStr)

            if data['death_yn'] == 1:
                trainMor[1] += 1
            else:
                trainMor[0] += 1
            if data['vasso_yn'] == 1:
                trainVas[1] += 1
            else:
                trainVas[0] += 1
            if data['intubation_yn'] == 1:
                trainInt[1] += 1
            else:
                trainInt[0] += 1
            if data['cpr_yn'] == 1:
                trainCpr[1] += 1
            else:
                trainCpr[0] += 1
    
    outputFile.close()
    testFile = open("builder/data/text/textDataset/" + datasetName + "_test_character_textDataset.txt", "w")

    with os.scandir(test_data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it, total=6839)):
            if "txt0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)

            TimeBetweenAdmVital = data['window_first_idx_hr'] - DatetimeToHours(data['admission_time'])
            if TimeBetweenAdmVital < 0 or TimeBetweenAdmVital > 21:
                continue

            endTime = 36 - TimeBetweenAdmVital
            startTime = -TimeBetweenAdmVital

            txtInput = data['txt_input']
            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")
            
            printStr = data['pat_id'] + " " + data['chid'] + " /"

            for char in txtInput:
                if char in vocabDict:
                    printStr += str(vocabDict[char]) + " "
                else:
                    printStr += "0 "
            
            printStr = printStr.strip()
            printStr += "/ "

            if data['death_yn'] == 1 and data['death_time'] is None:
                data['death_yn'] = 0
            elif data['death_yn'] == 1 and data['death_time'] > 96 - TimeBetweenAdmVital:
                data['death_yn'] = 0

            if data['death_yn'] == 1:
                #print(data['death_time'], type(data['death_time']))
                death_time = int(floor(float(data['death_time'])))
            else:
                death_time = None
            
            if data['vasso_yn'] == 1 and data['vasso_time'] is None:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1 and len(data['vasso_time']) < 1:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1:
                withinTime = False
                time = None
                for t in data['vasso_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['vasso_yn'] = 0

            if data['cpr_yn'] == 1 and data['cpr_time'] is None:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1 and len(data['cpr_time']) < 1:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1:
                withinTime = False
                time = None
                for t in data['cpr_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['cpr_yn'] = 0

            if data['intubation_yn'] == 1 and data['intubation_time'] is None:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1 and len(data['intubation_time']) < 1:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1:
                withinTime = False
                time = None
                for t in data['intubation_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['intubation_yn'] = 0
            
            printStr += str(data['death_yn'])
            if death_time is None:
                printStr += " -1"
            else:
                printStr += " " + str(death_time)

            printStr += " " + str(data['vasso_yn'])
            if data['vasso_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['cpr_yn'])
            if data['cpr_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['intubation_yn'])
            if data['intubation_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += "\n"

            if data['death_yn'] == 1:
                testMor[1] += 1
            else:
                testMor[0] += 1
            if data['vasso_yn'] == 1:
                testVas[1] += 1
            else:
                testVas[0] += 1
            if data['intubation_yn'] == 1:
                testInt[1] += 1
            else:
                testInt[0] += 1
            if data['cpr_yn'] == 1:
                testCpr[1] += 1
            else:
                testCpr[0] += 1

            testFile.write(printStr)
    testFile.close()

    print("Train Mortality : ", trainMor)
    print("Train Vasso : ", trainVas)
    print("Train Intubation : ", trainInt)
    print("Train Cpr : ", trainCpr)
    print("Test Mortality : ", testMor)
    print("Test Vasso : ", testVas)
    print("Test Intubation : ", testInt)
    print("Test Cpr : ", testCpr)

brokeChars = {'<', '：', '@', '有', 'ⅱ', '^', '#', '\x9d', '/', '[', ')', ',', '  '}
corpus = []


if token_type == "bpe":
    print("Token Type : BPE")
    with os.scandir(data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it)):
            if "txt0" in str(pkl_path):
                continue
            if "img0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            txtInput = data['txt_input']
            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")
            corpus.append(txtInput)
    
    # Mimic ICU / 38 Letters, 1600 Words
    # Mimic ED / 38 Letters, 3700 Words
    # SVRC ICU / 1127 Letters, 45000 Words
    vocabCount = 8000

    encoder = Encoder(vocabCount, pct_bpe = 0.88)
    encoder.fit(corpus)

    k = 0
    trainMor = [0,0]
    trainVas = [0,0]
    trainCpr = [0,0]
    trainInt = [0,0]
    testMor = [0,0]
    testVas = [0,0]
    testCpr = [0,0]
    testInt = [0,0]

    print("Training Dataset Synthesis")
    with os.scandir(data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it)):
            if "txt0" in str(pkl_path):
                continue
            if "img0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            
            TimeBetweenAdmVital = data['window_first_idx_hr'] - DatetimeToHours(data['admission_time'])
            if TimeBetweenAdmVital < 0 or TimeBetweenAdmVital > 21:
                continue
            print(TimeBetweenAdmVital)

            endTime = 36 - TimeBetweenAdmVital
            startTime = -TimeBetweenAdmVital

            k += 1

            txtInput = data['txt_input']
            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")

            if data['death_yn'] == 1 and data['death_time'] is None:
                data['death_yn'] = 0
            elif data['death_yn'] == 1 and data['death_time'] > 96 - TimeBetweenAdmVital:
                data['death_yn'] = 0

            if data['death_yn'] == 1:
                #print(data['death_time'], type(data['death_time']))
                death_time = int(floor(float(data['death_time'])))
            else:
                death_time = None
            
            # print(txtInput)
            # print(encoder.tokenize(txtInput))
            # print(next(encoder.transform([txtInput])))
            # print(next(encoder.inverse_transform(encoder.transform([txtInput]))))

            if data['vasso_yn'] == 1 and data['vasso_time'] is None:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1 and len(data['vasso_time']) < 1:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1:
                withinTime = False
                time = None
                for t in data['vasso_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['vasso_yn'] = 0

            if data['cpr_yn'] == 1 and data['cpr_time'] is None:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1 and len(data['cpr_time']) < 1:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1:
                withinTime = False
                time = None
                for t in data['cpr_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['cpr_yn'] = 0

            if data['intubation_yn'] == 1 and data['intubation_time'] is None:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1 and len(data['intubation_time']) < 1:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1:
                if datasetName == "sev_icu":
                    withinTime = False
                    time = None
                    for t in data['intubation_time']:
                        if t <= endTime and t >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t
                    if not withinTime:
                        data['intubation_yn'] = 0
                else:
                    withinTime = False
                    time = None
                    for t in data['intubation_time']:
                        if t[0] <= endTime and t[1] >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t[0]
                                if time < startTime:
                                    time = startTime
                                time = int(floor(time))
                    if not withinTime:
                        data['intubation_yn'] = 0

            printStr = data['pat_id'] + " " + data['chid'] + " /"
            tokens = next(encoder.transform([txtInput]))
            for token in tokens:
                if token in [0,1,2,3]:
                    printStr += str(token + vocabCount + 1) + " "
                else:
                    printStr += str(token) + " "
            printStr = printStr.strip() + "/ "

            printStr += str(data['death_yn'])
            if death_time is None:
                printStr += " -1"
            else:
                printStr += " " + str(death_time)

            printStr += " " + str(data['vasso_yn'])
            if data['vasso_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['cpr_yn'])
            if data['cpr_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['intubation_yn'])
            if data['intubation_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += "\n"

            outputFile.write(printStr)

            if data['death_yn'] == 1:
                trainMor[1] += 1
            else:
                trainMor[0] += 1
            if data['vasso_yn'] == 1:
                trainVas[1] += 1
            else:
                trainVas[0] += 1
            if data['intubation_yn'] == 1:
                trainInt[1] += 1
            else:
                trainInt[0] += 1
            if data['cpr_yn'] == 1:
                trainCpr[1] += 1
            else:
                trainCpr[0] += 1

    print("K: ", k)

    outputFile.close()
    testFile = open("builder/data/text/textDataset/" + datasetName + "_test_bpe_full_textDataset.txt", "w")

    with os.scandir(test_data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it)):
            if "txt0" in str(pkl_path):
                continue
            if "img0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            txtInput = data['txt_input']
            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")

            TimeBetweenAdmVital = data['window_first_idx_hr'] - DatetimeToHours(data['admission_time'])
            if TimeBetweenAdmVital < 0 or TimeBetweenAdmVital > 21:
                continue

            if data['death_yn'] == 1 and data['death_time'] is None:
                data['death_yn'] = 0
            elif data['death_yn'] == 1 and data['death_time'] > 96 - TimeBetweenAdmVital:
                data['death_yn'] = 0

            if data['death_yn'] == 1:
                #print(data['death_time'], type(data['death_time']))
                death_time = int(floor(float(data['death_time'])))
            else:
                death_time = None
            
            # print(txtInput)
            # print(encoder.tokenize(txtInput))
            # print(next(encoder.transform([txtInput])))
            # print(next(encoder.inverse_transform(encoder.transform([txtInput]))))

            if data['vasso_yn'] == 1 and data['vasso_time'] is None:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1 and len(data['vasso_time']) < 1:
                data['vasso_yn'] = 0
            elif['vasso_yn'] == 1:
                withinTime = False
                time = None
                for t in data['vasso_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['vasso_yn'] = 0

            if data['cpr_yn'] == 1 and data['cpr_time'] is None:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1 and len(data['cpr_time']) < 1:
                data['cpr_yn'] = 0
            elif['cpr_yn'] == 1:
                withinTime = False
                time = None
                for t in data['cpr_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['cpr_yn'] = 0

            if data['intubation_yn'] == 1 and data['intubation_time'] is None:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1 and len(data['intubation_time']) < 1:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1:
                if datasetName == "sev_icu":
                    withinTime = False
                    time = None
                    for t in data['intubation_time']:
                        if t <= endTime and t >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t
                    if not withinTime:
                        data['intubation_yn'] = 0
                else:
                    withinTime = False
                    time = None
                    for t in data['intubation_time']:
                        if t[0] <= endTime and t[1] >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t[0]
                                if time < startTime:
                                    time = startTime
                                time = int(floor(time))
                    if not withinTime:
                        data['intubation_yn'] = 0

            printStr = data['pat_id'] + " " + data['chid'] + " /"
            tokens = next(encoder.transform([txtInput]))
            for token in tokens:
                if token in [0,1,2,3]:
                    printStr += str(token + vocabCount + 1) + " "
                else:
                    printStr += str(token) + " "
            printStr = printStr.strip() + "/ "

            printStr += str(data['death_yn'])
            if death_time is None:
                printStr += " -1"
            else:
                printStr += " " + str(death_time)

            printStr += " " + str(data['vasso_yn'])
            if data['vasso_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['cpr_yn'])
            if data['cpr_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['intubation_yn'])
            if data['intubation_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += "\n"

            testFile.write(printStr)

            if data['death_yn'] == 1:
                testMor[1] += 1
            else:
                testMor[0] += 1
            if data['vasso_yn'] == 1:
                testVas[1] += 1
            else:
                testVas[0] += 1
            if data['intubation_yn'] == 1:
                testInt[1] += 1
            else:
                testInt[0] += 1
            if data['cpr_yn'] == 1:
                testCpr[1] += 1
            else:
                testCpr[0] += 1

    testFile.close()
    encoderFileHandler = open("builder/data/text/textDatasetEncoder/" + datasetName + "_" + str(vocabCount) + "_full.obj", 'wb')
    pkl.dump(encoder, encoderFileHandler)
    encoderFileHandler.close()

    print("Train Mortality : ", trainMor)
    print("Train Vasso : ", trainVas)
    print("Train Intubation : ", trainInt)
    print("Train Cpr : ", trainCpr)
    print("Test Mortality : ", testMor)
    print("Test Vasso : ", testVas)
    print("Test Intubation : ", testInt)
    print("Test Cpr : ", testCpr)

maxToken = 0
if token_type == "bert":
    max_length = 128

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # t = tokenizer("Hello World! Hello World!")
    # print(t)
    # print(tokenizer.decode(t['input_ids']))
    # print(tokenizer.decode([0, 1, 2, 3]))

    trainMor = [0,0]
    trainVas = [0,0]
    trainCpr = [0,0]
    trainInt = [0,0]
    testMor = [0,0]
    testVas = [0,0]
    testCpr = [0,0]
    testInt = [0,0]

    with os.scandir(data_path) as it:
        for idx, pkl_path in enumerate(tqdm(it)):
            if "txt0" in str(pkl_path):
                continue
            if "img0" in str(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)

            TimeBetweenAdmVital = data['window_first_idx_hr'] - DatetimeToHours(data['admission_time'])
            if TimeBetweenAdmVital < 0 or TimeBetweenAdmVital > 21:
                continue

            endTime = 36 - TimeBetweenAdmVital
            startTime = -TimeBetweenAdmVital
            
            txtInput = data['txt_input']
            
            # print(data)

            for char in brokeChars:
                txtInput = txtInput.replace(char, " ")
            
            if data['death_yn'] == 1 and data['death_time'] is None:
                data['death_yn'] = 0
            elif data['death_yn'] == 1 and data['death_time'] > 96 - TimeBetweenAdmVital:
                data['death_yn'] = 0

            if data['death_yn'] == 1:
                #print(data['death_time'], type(data['death_time']))
                death_time = int(floor(float(data['death_time'])))
            else:
                death_time = None
            
            if data['vasso_yn'] == 1 and data['vasso_time'] is None:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1 and len(data['vasso_time']) < 1:
                data['vasso_yn'] = 0
            elif data['vasso_yn'] == 1:
                withinTime = False
                time = None
                for t in data['vasso_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['vasso_yn'] = 0

            if data['cpr_yn'] == 1 and data['cpr_time'] is None:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1 and len(data['cpr_time']) < 1:
                data['cpr_yn'] = 0
            elif data['cpr_yn'] == 1:
                withinTime = False
                time = None
                for t in data['cpr_time']:
                    if t[0] <= endTime and t[1] >= startTime:
                        withinTime = True
                        if time is not None:
                            time = t[0]
                            if time < startTime:
                                time = startTime
                            time = int(floor(time))
                if not withinTime:
                    data['cpr_yn'] = 0

            if data['intubation_yn'] == 1 and data['intubation_time'] is None:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1 and len(data['intubation_time']) < 1:
                data['intubation_yn'] = 0
            elif data['intubation_yn'] == 1:
                if datasetName == "sev_icu":
                    withinTime = False
                    time = None
                    for t in data['intubation_time']:
                        if t <= endTime and t >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t
                    if not withinTime:
                        data['intubation_yn'] = 0
                else:
                    withinTime = False
                    time = None
                    for t in data['intubation_time']:
                        if t[0] <= endTime and t[1] >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t[0]
                                if time < startTime:
                                    time = startTime
                                time = int(floor(time))
                    if not withinTime:
                        data['intubation_yn'] = 0

            printStr = data['pat_id'] + " " + data["chid"] + " /"
            tokens = tokenizer(txtInput)['input_ids'][1:-1]

            for token in tokens:
                printStr += str(token) + " "
                if token > maxToken:
                    maxToken = token
            
            printStr = printStr.strip() + "/ "

            printStr += str(data['death_yn'])
            if death_time is None:
                printStr += " -1"
            else:
                printStr += " " + str(death_time)

            printStr += " " + str(data['vasso_yn'])
            if data['vasso_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['cpr_yn'])
            if data['cpr_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += " " + str(data['intubation_yn'])
            if data['intubation_yn'] == 1:
                printStr += " " + str(time)
            else:
                printStr += " -1"
            printStr += "\n"

            outputFile.write(printStr)

            if data['death_yn'] == 1:
                trainMor[1] += 1
            else:
                trainMor[0] += 1
            if data['vasso_yn'] == 1:
                trainVas[1] += 1
            else:
                trainVas[0] += 1
            if data['intubation_yn'] == 1:
                trainInt[1] += 1
            else:
                trainInt[0] += 1
            if data['cpr_yn'] == 1:
                trainCpr[1] += 1
            else:
                trainCpr[0] += 1

        
        outputFile.close()
        testFile = open("builder/data/text/textDataset/" + datasetName + "_test_bert_full_textDataset.txt", "w")

        with os.scandir(test_data_path) as it:
            for idx, pkl_path in enumerate(tqdm(it)):
                if "txt0" in str(pkl_path):
                    continue
                if "img0" in str(pkl_path):
                    continue
                with open(pkl_path, 'rb') as f:
                    data = pkl.load(f)
                txtInput = data['txt_input']
                for char in brokeChars:
                    txtInput = txtInput.replace(char, " ")

                TimeBetweenAdmVital = data['window_first_idx_hr'] - DatetimeToHours(data['admission_time'])
                if TimeBetweenAdmVital < 0 or TimeBetweenAdmVital > 21:
                    continue

                if data['death_yn'] == 1 and data['death_time'] is None:
                    data['death_yn'] = 0
                elif data['death_yn'] == 1 and data['death_time'] > 96 - TimeBetweenAdmVital:
                    data['death_yn'] = 0

                if data['death_yn'] == 1:
                    #print(data['death_time'], type(data['death_time']))
                    death_time = int(floor(float(data['death_time'])))
                else:
                    death_time = None
                
                if data['vasso_yn'] == 1 and data['vasso_time'] is None:
                    data['vasso_yn'] = 0
                elif data['vasso_yn'] == 1 and len(data['vasso_time']) < 1:
                    data['vasso_yn'] = 0
                elif['vasso_yn'] == 1:
                    withinTime = False
                    time = None
                    for t in data['vasso_time']:
                        if t[0] <= endTime and t[1] >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t[0]
                                if time < startTime:
                                    time = startTime
                                time = int(floor(time))
                    if not withinTime:
                        data['vasso_yn'] = 0

                if data['cpr_yn'] == 1 and data['cpr_time'] is None:
                    data['cpr_yn'] = 0
                elif data['cpr_yn'] == 1 and len(data['cpr_time']) < 1:
                    data['cpr_yn'] = 0
                elif['cpr_yn'] == 1:
                    withinTime = False
                    time = None
                    for t in data['cpr_time']:
                        if t[0] <= endTime and t[1] >= startTime:
                            withinTime = True
                            if time is not None:
                                time = t[0]
                                if time < startTime:
                                    time = startTime
                                time = int(floor(time))
                    if not withinTime:
                        data['cpr_yn'] = 0

                if data['intubation_yn'] == 1 and data['intubation_time'] is None:
                    data['intubation_yn'] = 0
                elif data['intubation_yn'] == 1 and len(data['intubation_time']) < 1:
                    data['intubation_yn'] = 0
                elif data['intubation_yn'] == 1:
                    if datasetName == "sev_icu":
                        withinTime = False
                        time = None
                        for t in data['intubation_time']:
                            if t <= endTime and t >= startTime:
                                withinTime = True
                                if time is not None:
                                    time = t
                        if not withinTime:
                            data['intubation_yn'] = 0
                    else:
                        withinTime = False
                        time = None
                        for t in data['intubation_time']:
                            if t[0] <= endTime and t[1] >= startTime:
                                withinTime = True
                                if time is not None:
                                    time = t[0]
                                    if time < startTime:
                                        time = startTime
                                    time = int(floor(time))
                        if not withinTime:
                            data['intubation_yn'] = 0

                printStr = data['pat_id'] + " " + data['chid'] + " /"
                tokens = tokenizer(txtInput)['input_ids'][1:-1]
                for token in tokens:
                    printStr += str(token) + " "
                    if token > maxToken:
                        maxToken = token
                printStr = printStr.strip() + "/ "

                printStr += str(data['death_yn'])
                if death_time is None:
                    printStr += " -1"
                else:
                    printStr += " " + str(death_time)

                printStr += " " + str(data['vasso_yn'])
                if data['vasso_yn'] == 1:
                    printStr += " " + str(time)
                else:
                    printStr += " -1"
                printStr += " " + str(data['cpr_yn'])
                if data['cpr_yn'] == 1:
                    printStr += " " + str(time)
                else:
                    printStr += " -1"
                printStr += " " + str(data['intubation_yn'])
                if data['intubation_yn'] == 1:
                    printStr += " " + str(time)
                else:
                    printStr += " -1"
                printStr += "\n"

                testFile.write(printStr)

                if data['death_yn'] == 1:
                    testMor[1] += 1
                else:
                    testMor[0] += 1
                if data['vasso_yn'] == 1:
                    testVas[1] += 1
                else:
                    testVas[0] += 1
                if data['intubation_yn'] == 1:
                    testInt[1] += 1
                else:
                    testInt[0] += 1
                if data['cpr_yn'] == 1:
                    testCpr[1] += 1
                else:
                    testCpr[0] += 1

    testFile.close()
    encoderFileHandler = open("builder/data/text/textDatasetEncoder/" + datasetName + "_full.obj", 'wb')
    pkl.dump(tokenizer, encoderFileHandler)
    encoderFileHandler.close()
    
    print("Train Mortality : ", trainMor)
    print("Train Vasso : ", trainVas)
    print("Train Intubation : ", trainInt)
    print("Train Cpr : ", trainCpr)
    print("Test Mortality : ", testMor)
    print("Test Vasso : ", testVas)
    print("Test Intubation : ", testInt)
    print("Test Cpr : ", testCpr)