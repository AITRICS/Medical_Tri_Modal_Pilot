


# inputFileNm = "builder/data/text/textDataset/mimic_icu_test_bert_textDataset.txt"
# inputFile = open(inputFileNm, "r")

# maxToken = 0
# tokenDict = {}
# while True:
#     line = inputFile.readline()
#     if not line:
#         break
#     tokens = [int(x) for x in line.strip().split("/")[1].split()]
#     for token in tokens:
#         if token > maxToken:
#             maxToken = token
#         if token in tokenDict:
#             tokenDict[token] += 1
#         else:
#             tokenDict[token] = 1

# print("Max Token : ", maxToken)
# print("Token Count : ", len(tokenDict))

# inputFile.close()

import os
import pickle as pkl
from tqdm import tqdm

train_data_path = "/nfs/thena/shared/multi_modal/training_data_0706/sev_icu/train"
test_data_path = "/nfs/thena/shared/multi_modal/training_data_0706/sev_icu/test"

with os.scandir(train_data_path) as it:
    for idx, pkl_path in enumerate(tqdm(it)):
        if "txt0" in str(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
        
        txtInput = data['txt_input']
        if idx < 1000 and data['intubation_yn'] == 1:
            print(data['intubation_time'])