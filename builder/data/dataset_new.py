# Copyright (c) 2022, Kwanhyung Lee AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import math
import torch
import random
#import pickle
import pickle as pkl
import pickle5 as pickle
# import pickle5 as pickle
import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t
from control.config import args
from builder.utils.utils import *
from builder.data.collate_fn import *

VITALSIGN_LABTEST = ['HR', 'RR', 'BT', 'SBP', 'DBP', 'Sat', 'GCS', 
                     'Hematocrit', 'PLT', 'WBC', 'Bilirubin', 'pH', 'HCO3', 
                     'Creatinine', 'Lactate', 'Potassium', 'Sodium', 'CRP']

#margin_dir = search_walk({"path": "/nfs/thena/MedicalAI/ImageData/public/MIMIC_CXR/data/files_margins", "extension": ".jpg"})
#with open("margin_dir.pkl","wb") as f:
#    pickle.dump(margin_dir,f)

# with open("margin_dir.pkl","rb") as f:
#     margin_dir = pickle.load(f)

######################################################################################
################################ Preprocess functions ################################
######################################################################################
# def resize(matching):
#     size = 256
#     aspect_ratio_num = float(re.findall("\d[.]\d+",matching[0])[0])
#     if aspect_ratio_num > 1.0: #논문에서 비율 유지
#         resize = (int(size*aspect_ratio_num), size)
#     elif aspect_ratio_num == 1.0:
#         resize = (size, size)
#     else:
#         resize = (size, int(size*1/aspect_ratio_num))
#     return resize 

def DatetimeToHours(time):
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    return (year * 8760) + (month * 730) + (day * 24) + hour + (minute / float(60))

def xray_image_transform_train():
    # transform (load jpeg img, add channel, rescale 0~1, random rotation)
    """
    MONAI
    transform = transforms.Compose([
                transforms.LoadImage(image_only=True, reader=PILReader()),
                transforms.AddChannel(),
                transforms.Resize(resize(matching), mode="bilinear"),##### resize 시간이 오래 걸리는 것은 아닌지
                transforms.RandSpatialCrop((224,224), random_size=False),#### OR
                transforms.CenterSpatialCrop((224,224), random_size=False),####
                transforms.ScaleIntensity(),
                #transforms.RandRotate(range_x=5 * (np.pi / 180), padding_mode='zeros', 
                #                        prob=0.5, keep_size=True),   # (-5, 5) degree --> radian
                transforms.ToTensor(),
                ])
    """
    transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.1), ratio=(3/4, 4/3)),
                transforms.ToTensor(),
                ])  
    
    return transform


def xray_image_transform_center_val():
    # transform (load jpeg img, add channel, rescale 0~1, random rotation)
    """
    MONAI
    transform = transforms.Compose([
                transforms.LoadImage(image_only=True, reader=PILReader()),
                transforms.AddChannel(),
                transforms.Resize(resize(matching), mode="bilinear"),#####
                transforms.CenterSpatialCrop((224,224)),
                transforms.ScaleIntensity(),
                transforms.ToTensor(),
                ])
    """
    transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),   
                transforms.ToTensor(),
                ])
    
    return transform

def xray_image_transform_resize_val():
    # transform (load jpeg img, add channel, rescale 0~1, random rotation)
    transform = transforms.Compose([
                transforms.Resize((args.image_size,args.image_size)),   
                transforms.ToTensor(),
                ])
    
    return transform


def clinical_note_transform(tokens):
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
            
    return tokens

# CUDA_VISIBLE_DEVICES=7 python ./2_train.py --project-name test --model gru_d --optim adam --epoch 75 --batch-size 16 --cross-fold-val True --lr-scheduler Single --input-types vslt --output-type mortality --predict-type binary --modality-inclusion fullmodal
### ! dataset can start with partially available 생체신호
### ! for later low accuracy, we can change the starting point to where all 1 in the mask
class Onetime_Outbreak_Training_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Training Dataset for Onetime_Outbreak_Dataset Prediction...")
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        patient_list = []
        
        self.feature_mins = None
        self.feature_maxs = None
        self.transform = None
        self.txtDict = None
        
        lengths = []
        positive_tpoints = 0
        negative_tpoints = 0
        
        self.vslt_len = len(args.vitalsign_labtest)
        self.neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.intv_len = int(args.prediction_range // 12)
        self.window_size = args.window_size
        self.image_size = [args.image_size, args.image_size]
        self.input_types = args.input_types
        self.image_data_path = args.image_data_path
        self.token_max_length = args.bert_token_max_length
        self.txtDict = txtDictLoad("train")
        
        class2dict_missing = {3:1, 6:2, 9:3, 2:4, 8:6, 11:7, 1:4, 4:5, 7:6, 10:7}
        class2dict_full = {2:0}
                
        # real-time x-ray image transform function
        if ("img" in self.input_types or 'train-missing' in args.modality_inclusion):
            self.transform = xray_image_transform_train()
                
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                
            if "cxr_input" in data_info:
                new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
                if len(new_cxr_inputs) > 0:
                    data_info["cxr_input"] = new_cxr_inputs
                else:
                    del data_info['cxr_input']
                    file_name = file_name.replace("_img1", "_img0")
                    
            if 'train-full' in args.modality_inclusion:
                if args.fullmodal_definition not in file_name:
                    continue
                if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                    continue
                if "txt1" in args.fullmodal_definition:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        continue
            else: # missing modality
                if "txt1" in file_name:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        file_name = file_name.replace("_txt1_", "_txt0_")
                if ("cxr_input" not in data_info and "img1" in file_name):
                    file_name = file_name.replace("_img1", "_img0")
                
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue            
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                continue
                
            ##### get possibles indices and max lengths here #####
            # If patient hasn't died, sample random sequence of 24 hours
            # target_type = 0: non patient
            # target_type = 1: patient with outbreak soon
            # target_type = 2: patient with outbreak on far future or far past but currently in normal group
            #
            # target = 0: normal (negative)
            # target = 1: abnormal (positive)
            #
            # modality_target_type = 1~12 (1: vslt_pp, 2: vslt_pn, 3: vslt_nn, 4: vslt+txt_pp, 5: vslt+txt_pn, 6: vslt+txt_nn, 
            #                               7: vslt+img_pp, 8: vslt+img_pn, 9: vslt+img_nn, 10: vslt+txt+img_pp, 11: vslt+txt+img_pn, 12: vslt+txt+img_nn)
            possible_indices_dict = {}
            # possible_indices_keys: 0 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pp, nn)
            # possible_indices_keys_with_img: 1 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pp, nn)     Case3: (wimgwtxt_pp: 0, wimgwtxt-nn: 2, wimgw/otxt_pp: 3, wimgw/otxt-nn: 5)
            # possible_indices_keys_without_img: 2 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt_pp: 6, w/oimgwtxt-nn: 8, w/oimgw/otxt_pp: 9, w/oimgw/otxt-nn: 11)
            # pat_neg_indices_keys = 3 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pn)
            # pat_neg_indices_keys_with_img = 4 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pn)         Case3: (wimgwtxt-pn: 1, wimgw/otxt-pn: 4)
            # pat_neg_indices_keys_without_img = 5 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt-pn: 7, w/oimgw/otxt-pn: 10)
            possible_indices_keys_alltypes = [[] for _ in range(6)]
            
            if (data_info['death_yn'] == 0):
                target = 0
                target_type = 0

                possible_indices_keys_alltypes[0] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
                    
            else:
                death_time = data_info['death_time']
                # If death time is beyond the prediction range of the given data or happened too early, change to 0 target
                if (death_time > sequenceLength + args.prediction_range - 1) or (death_time < args.min_inputlen):
                    target = 0
                    target_type = 2
                    
                    possible_indices_keys_alltypes[3] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
                    
                else:              
                    target = 1
                    target_type = 1
                    # For within-n-hours task, the data must be within prediction_range of death time
                    # range 2~3 means 2 < x =< 3
                    death_time = math.ceil(death_time)
                    possible_indices = [(death_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (death_time >= args.min_inputlen+i-1) and (death_time - i < sequenceLength)] or None
                    
                    if possible_indices is None:
                        print("SeqLength : " + str(sequenceLength))
                        print(data_info)
                        raise Exception('Classification Error')
                
                    for p_index in possible_indices:
                        if p_index[0] not in possible_indices_dict:
                            possible_indices_dict[p_index[0]] = []
                        if p_index[1] not in possible_indices_dict[p_index[0]]:
                            possible_indices_dict[p_index[0]].append(p_index[1])
                        if p_index[0] not in possible_indices_keys_alltypes[0]:
                            possible_indices_keys_alltypes[0].append(p_index[0])
                    possible_indices_keys_alltypes[0].sort()

            if target_type in [0, 1]:
                if (("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                    earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                    possible_indices_keys_alltypes[1]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time<=i])    
                    possible_indices_keys_alltypes[2]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time>i])    
                elif ('train-missing' in args.modality_inclusion):
                    possible_indices_keys_alltypes[2]= list(possible_indices_keys_alltypes[0])
            
            if ("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion and target_type != 2):
                if not possible_indices_keys_alltypes[1]: 
                    continue
            
            if target == 1 or target_type == 2:
                if target == 1:
                    all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)]
                    possible_indices_keys_alltypes[3] = [item for item in all_indices_keys if item not in possible_indices_keys_alltypes[0]]

                if ('train-full' in args.modality_inclusion and "img1" in args.fullmodal_definition) or ('train-missing' in args.modality_inclusion) and ('cxr_input' in data_info):
                    earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                    possible_indices_keys_alltypes[4]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time<=i])
                    possible_indices_keys_alltypes[5]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time>i])    
                elif ('train-missing' in args.modality_inclusion):
                    possible_indices_keys_alltypes[5]= list(possible_indices_keys_alltypes[3])    
            
            possibleWinSizes = data_info['possibleWinSizes']
            # possible_indices_keys = [key for key in possible_indices_keys if key in possibleWinSizes]
            possible_indices_keys_alltypes = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices_keys_alltypes])
            
            if isListEmpty(possible_indices_keys_alltypes):
                continue
            patient_list.append(target_type)
            
                # elif ("txt1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion):
                #     pass
            
            ######################################################
            if ('train-full' in args.modality_inclusion and "img1" not in args.fullmodal_definition): # (Case1: full_modal with img1 not in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [0, 3]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type == 0:               
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:# possible_indices_keys가 빈 리스트라면 실행 안됨
                            self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, possibleWinSizes, target])
                            self._type_list.append(target_type)
                            
                            possible_tpoints = [True if i in possible_indices_dict else False for i in possible_indices_keys]
                            positive_tpoints += possible_tpoints.count(True)
                            negative_tpoints += possible_tpoints.count(False)
                    else:                    
                        if target == 1 and len(possible_indices_keys) > 0 and possible_indices_keys is not None:
                            self._data_list.append([pkl_path, possible_indices_keys, {}, possibleWinSizes, 0])
                            self._type_list.append(2)
                            negative_tpoints += len(possible_indices_keys)
                                
            elif ('train-full' in args.modality_inclusion and "img1" in args.fullmodal_definition): # (Case2: full_modal with img1 in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 4]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type == 0:               
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:# possible_indices_keys가 빈 리스트라면 실행 안됨
                            self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, possibleWinSizes, target])
                            self._type_list.append(target_type)
                            
                            possible_tpoints = [True if i in possible_indices_dict else False for i in possible_indices_keys]
                            positive_tpoints += possible_tpoints.count(True)
                            negative_tpoints += possible_tpoints.count(False)
                    else:                    
                        if target == 1 and len(possible_indices_keys) > 0 and possible_indices_keys is not None:
                            self._data_list.append([pkl_path, possible_indices_keys, {}, possibleWinSizes, 0])
                            self._type_list.append(2)
                            negative_tpoints += len(possible_indices_keys)
                            
            else: # (Case3: missing modal)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 2, 4, 5]])
                
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type < 2:               
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:# possible_indices_keys가 빈 리스트라면 실행 안됨
                            self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, possibleWinSizes, target])
                            if keylist_type == 0 and target_type == 1 and "txt1" in file_name:
                                self._type_list.append(0)
                            elif keylist_type == 0 and target_type == 0 and "txt1" in file_name:
                                self._type_list.append(2)
                            elif keylist_type == 0 and target_type == 1 and "txt1" not in file_name:
                                self._type_list.append(3)
                            elif keylist_type == 0 and target_type == 0 and "txt1" not in file_name:
                                self._type_list.append(5)
                            elif keylist_type == 1 and target_type == 1 and "txt1" in file_name:
                                self._type_list.append(6)
                            elif keylist_type == 1 and target_type == 0 and "txt1" in file_name:
                                self._type_list.append(8)
                            elif keylist_type == 1 and target_type == 1 and "txt1" not in file_name:
                                self._type_list.append(9)
                            elif keylist_type == 1 and target_type == 0 and "txt1" not in file_name:
                                self._type_list.append(11)
                            else:
                                print("Missing modal error with keylist_type < 2")
                                exit(1)
                            possible_tpoints = [True if i in possible_indices_dict else False for i in possible_indices_keys]
                            positive_tpoints += possible_tpoints.count(True)
                            negative_tpoints += possible_tpoints.count(False)
                    else:                    
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:
                            self._data_list.append([pkl_path, possible_indices_keys, {}, possibleWinSizes, 0])
                            if keylist_type == 2 and "txt1" in file_name:
                                self._type_list.append(1)
                            elif keylist_type == 2 and "txt1" not in file_name:
                                self._type_list.append(4)
                            elif keylist_type == 3 and "txt1" in file_name:
                                self._type_list.append(7)
                            elif keylist_type == 3 and "txt1" not in file_name:
                                self._type_list.append(10)
                            else:
                                print("Missing modal error with keylist_type >= 2")
                                exit(1)
                                # print("### 2 ###")
                                # print("keylist_type: ", keylist_type)
                                # print("file_name: ", file_name)

                            negative_tpoints += len(possible_indices_keys)
                
                
            ######################################################            
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)

        if ('train-full' in args.modality_inclusion):
            ### class 2 방식
            self._type_list = [class2dict_full[i] if i in class2dict_full else i for i in self._type_list]
        else:
            ### class 2 방식
            self._type_list = [class2dict_missing[i] if i in class2dict_missing else i for i in self._type_list]
            
        self.feature_means = list(data_info['mean'])
        print("No Dataset Error: ", len(self._type_list) == len(self._data_list))
        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
        if ('train-full' in args.modality_inclusion):
            print("Number of patient positive samples list for training: {}".format(str(patient_list.count(1))))
            print("Number of patient negative samples list for training: {}".format(str(patient_list.count(2))))
            print("Number of non-patient negative samples list for training: {}".format(str(patient_list.count(0))))        
            print("Number of total negative samples list for training: {}".format(str(patient_list.count(0) + patient_list.count(2))))        
        else: # missing modality
            print("[1]. Number of patients: ", patient_list.count(1))
            print("1. Number of patient positive samples lists for training", self._type_list.count(0)+self._type_list.count(1)+self._type_list.count(2)+self._type_list.count(3))
            print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
            print(" 1-2. wimg-w/otxt_pp: ", self._type_list.count(1))
            print(" 1-3. w/oimg-wtxt_pp: ", self._type_list.count(2))
            print(" 1-4. w/oimg-w/otxt_pp: ", self._type_list.count(3))
            print("[2]. Number of patients with negative signs: ", patient_list.count(2))
            # print("2. Number of patient negative samples list for training", self._type_list.count(8)+self._type_list.count(9)+self._type_list.count(10)+self._type_list.count(11))
            # print(" 2-1. wimg-wtxt-pn: ", self._type_list.count(8))
            # print(" 2-2. wimg-w/otxt-pn: ", self._type_list.count(9))
            # print(" 2-3. w/oimg-wtxt-pn: ", self._type_list.count(10))
            # print(" 2-4. w/oimg-w/otxt-pn: ", self._type_list.count(11))
            print("[3]. Number of non-patients: ", patient_list.count(0))
            print("3. Number of non-patient negative samples lists for training", self._type_list.count(4)+self._type_list.count(5)+self._type_list.count(6)+self._type_list.count(7))
            print(" 3-1. wimg-wtxt-nn: ", self._type_list.count(4))
            print(" 3-2. wimg-w/otxt-nn: ", self._type_list.count(5))
            print(" 3-3. w/oimg-wtxt-nn: ", self._type_list.count(6))
            print(" 3-4. w/oimg-w/otxt-nn: ", self._type_list.count(7))
        print("########## Detail Data Info ##########")
        print("Positive time-points: ", positive_tpoints)
        print("Negative time-points: ", negative_tpoints)
        
        self.feature_mins = self.train_min
        self.feature_maxs = self.train_max
        
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        pkl_path, possible_indices_keys, labels_by_dict, win_sizes, target = self._data_list[index]
        type_list = self._type_list[index]
        
        file_name = pkl_path.split("/")[-1]
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

        windowIndex = self.window_size - 1
        
        # if 'train-full' in args.modality_inclusion:
        #     earliest_img_time = min([j[0] for j in data_pkl['cxr_input']])
        #     possible_indices_keys= list([i for i in possible_indices_keys if earliest_img_time<=i])
        
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey])
        
        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
            f_indices = False
        else:
            dataSequence, maskSequence, deltaSequence, inputLength, f_indices = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex+12, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-randLength) -1
                
                if (i <= selectedKey) and (i > selectedKey-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                    
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, self.window_size, self.vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = self.neg_multi_target
        else:
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // self.intv_len)
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)
        multi_target = torch.tensor(multi_target)
        
        missing = [False]   # Missing modality list: [vital/lab, img, txt]
        
        if (("img" in args.input_types and "img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion and type_list in [0,2,3,5] and "img" in args.input_types)) and ('cxr_input' in data_pkl):
            cxr_li = [i for i in data_pkl['cxr_input'] if i[0] <= selectedKey] 
            if not cxr_li and ('train-full' in args.modality_inclusion): 
                print("collate cxr error")
                exit(1)
            elif not cxr_li and ('train-missing' in args.modality_inclusion): 
                img = torch.zeros(self.image_size).unsqueeze(0)
                missing.append(True)
            else:
                cxr_path = sorted(cxr_li)[-1][1]
                image = Image.open(self.image_data_path + cxr_path)
                image = F_t.equalize(image)
                img = self.transform(image)
                missing.append(False)
        else:
            img = torch.zeros(self.image_size).unsqueeze(0)
            missing.append(True)
            
        if (("txt" in args.input_types and "txt1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion and "txt" in args.input_types)) and ("txt1" in file_name):
            tokens = self.txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
            if len(tokens) == 0:
                tokens = torch.zeros(self.token_max_length)
                textLength = 0
                missing.append(True)
            else:
                textLength = len(tokens)
                # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
                # EX) 2 {Sentence Tokens} {1 Padding} 3
                # Add Beginnning of Sentence Token
                tokens.insert(0, 2)
                tokens = torch.Tensor(clinical_note_transform(tokens))
                tokens[tokens==1] = 0
                missing.append(False)
        else:    
            tokens = torch.zeros(self.token_max_length)
            textLength = 0
            missing.append(True)
        missing = torch.Tensor(missing)
        return final_seqs, static_inputs, multi_target, inputLength, img, tokens, textLength, missing, f_indices

class Onetime_Outbreak_Test_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preparing {} Dataset for Onetime_Outbreak_Dataset Prediction...".format(data_type))
        self._data_list = []
        _data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        _type_list = []
        patient_list = []
        patDict = {}
        winDict = {}
        
        self.feature_mins = None
        self.feature_maxs = None
        self.transform = None
        self.txtDict = None
        
        lengths = []
        positive_tpoints = 0
        negative_tpoints = 0
        
        self.vslt_len = len(args.vitalsign_labtest)
        self.neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.intv_len = int(args.prediction_range // 12)
        self.window_size = args.window_size
        self.image_size = [args.image_size, args.image_size]
        self.input_types = args.input_types
        self.image_data_path = args.image_data_path
        self.token_max_length = args.bert_token_max_length
        
        class2dict_missing = {3:1, 6:2, 9:3, 2:4, 8:6, 11:7, 1:4, 4:5, 7:6, 10:7}
        class2dict_full = {2:0}
        
        load_flag = False

        test_index_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + ".pkl"
        test_winsize_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + "_winsize.pkl"
        validation_index_file = "./data/testIndexes/valIndexes__" + args.train_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + ".pkl"
        validation_winsize_file = "./data/testIndexes/valIndexes__" + args.train_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + "_winsize.pkl"
        
        if not os.path.exists("./data/testIndexes"):
            os.makedirs('./data/testIndexes', exist_ok=True)
            
        if data_type == "validation dataset":
            self.txtDict = txtDictLoad("train")
        else:
            self.txtDict = txtDictLoad("test")

        if data_type == "test dataset":
            if  os.path.exists(test_index_file) and os.path.exists(test_winsize_file) and data_type == "test dataset":
                # Open the file and add existing entries to dictionary
                print("Index file exists... Loading...")
                load_flag = True
                with open(test_index_file, 'rb') as f:
                    patDict = pickle.load(f)
                with open(test_winsize_file, 'rb') as f:
                    winDict = pickle.load(f)
            else:
                if os.path.exists(test_index_file):
                    os.remove(test_index_file)
                if os.path.exists(test_winsize_file):
                    os.remove(test_winsize_file)
                
        elif data_type == "validation dataset":
            if  os.path.exists(validation_index_file) and os.path.exists(validation_winsize_file) and data_type == "validation dataset":
                # Open the file and add existing entries to dictionary
                print("Index file exists... Loading...")
                load_flag = True
                with open(validation_index_file, 'rb') as f:
                    patDict = pickle.load(f)
                with open(validation_winsize_file, 'rb') as f:
                    winDict = pickle.load(f)
            else:
                print("Index file not exists... Generating...")
                if os.path.exists(validation_index_file):
                    os.remove(validation_index_file)
                if os.path.exists(validation_winsize_file):
                    os.remove(validation_winsize_file)    
                
                onetime_outbreak_valdataset_maker(args, validation_index_file, validation_winsize_file)
                with open(validation_index_file, 'rb') as f:
                    patDict = pickle.load(f)
                with open(validation_winsize_file, 'rb') as f:
                    winDict = pickle.load(f)
                    
        else:
            raise Exception('Data Type Error')
                
        # real-time x-ray image transform function
        if "img" in self.input_types:
            if args.image_test_type == "center":
                self.transform = xray_image_transform_center_val()
            elif args.image_test_type == "resize":
                self.transform = xray_image_transform_resize_val()

            # if args.image_size == 224:
            #     if args.image_test_type == "center":
            #         self.transform = xray_image_transform_224_center_val()
            #     elif args.image_test_type == "resize":
            #         self.transform = xray_image_transform_224_resize_val()
            #     else:
            #         print("You need to choose 'center' or 'resize' as image_test_type")
            # elif args.image_size == 512:
            #     if args.image_test_type == "center":
            #         self.transform = xray_image_transform_512_center_val()
            #     elif args.image_test_type == "resize":
            #         self.transform = xray_image_transform_512_resize_val()
            #     else:
            #         print("You need to choose 'center' or 'resize' as image_test_type")
            # else:
            #     print("You need to choose either 224 or 512 as image_size")
                

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            if "cxr_input" in data_info:
                new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
                if len(new_cxr_inputs) > 0:
                    data_info["cxr_input"] = new_cxr_inputs
                else:
                    del data_info['cxr_input']
                    file_name = file_name.replace("_img1", "_img0")
                
            if ('test-full' in args.modality_inclusion and data_type == "validation dataset") or ('test-full' in args.modality_inclusion and data_type == "test dataset"):
                if args.fullmodal_definition not in file_name:
                    continue
                if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                    continue
                if "txt1" in args.fullmodal_definition:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        continue
            else: # missing modality
                if "txt1" in file_name:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        file_name = file_name.replace("_txt1_", "_txt0_")
                if ("cxr_input" not in data_info and "img1" in file_name):
                    file_name = file_name.replace("_img1", "_img0")
            
            pat_id = int(data_info['pat_id'])
            chid = int(data_info['chid'])
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                continue
                
            # Check if the randIndex for the given patient has already been initialized
            if (pat_id, chid) in patDict:
                possible_indices_keys_alltypes, possible_indices_dict, target, possibleWinSizes, target_type = patDict[(pat_id, chid)]
                
                if isListEmpty(possible_indices_keys_alltypes):
                    continue
                
                # _data_list.append([pkl_path, possible_indices_keys_alltypes, possible_indices_dict, possibleWinSizes, target])
                # _type_list.append(target_type)
                                
            else:
                if data_type == "validation dataset":
                    continue
                ##### get possibles indices and max lengths here #####
                # If patient hasn't died, sample random sequence of 24 hours
                # target_type = 0: non patient
                # target_type = 1: patient with outbreak soon
                # target_type = 2: patient with outbreak on far future or far past but currently in normal group
                #
                # target = 0: normal (negative)
                # target = 1: abnormal (positive)
                #
                # modality_target_type = 1~12 (1: vslt_pp, 2: vslt_pn, 3: vslt_nn, 4: vslt+txt_pp, 5: vslt+txt_pn, 6: vslt+txt_nn, 
                #                               7: vslt+img_pp, 8: vslt+img_pn, 9: vslt+img_nn, 10: vslt+txt+img_pp, 11: vslt+txt+img_pn, 12: vslt+txt+img_nn)
                possible_indices_dict = {}
                # possible_indices_keys: 0 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pp, nn)
                # possible_indices_keys_with_img: 1 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pp, nn)     Case3: (wimgwtxt_pp: 0, wimgwtxt-nn: 2, wimgw/otxt_pp: 3, wimgw/otxt-nn: 5)
                # possible_indices_keys_without_img: 2 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt_pp: 6, w/oimgwtxt-nn: 8, w/oimgw/otxt_pp: 9, w/oimgw/otxt-nn: 11)
                # pat_neg_indices_keys = 3 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pn)
                # pat_neg_indices_keys_with_img = 4 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pn)         Case3: (wimgwtxt-pn: 1, wimgw/otxt-pn: 4)
                # pat_neg_indices_keys_without_img = 5 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt-pn: 7, w/oimgw/otxt-pn: 10)
                possible_indices_keys_alltypes = [[] for _ in range(6)]
                
                possibleWinSizes = data_info['possibleWinSizes']
                if(data_info['death_yn'] == 0):
                    target = 0
                    target_type = 0
                    
                    possible_indices_keys_alltypes[0] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
            
                else:
                    death_time = data_info['death_time']
                    # If death time is beyond the prediction range of the given data or happened too early, change to 0 target
                    if (death_time > sequenceLength + args.prediction_range - 1) or (death_time < args.min_inputlen):
                        target = 0
                        target_type = 2
                        
                        possible_indices_keys_alltypes[3] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
                        
                    else:              
                        target = 1
                        target_type = 1
                        # For within-n-hours task, the data must be within prediction_range of death time
                        # range 2~3 means 2 < x =< 3
                        death_time = math.ceil(death_time)
                        possible_indices = [(death_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (death_time >= args.min_inputlen+i-1) and (death_time - i < sequenceLength)] or None
                        
                        if possible_indices is None:
                            print("SeqLength : " + str(sequenceLength))
                            print(data_info)
                            raise Exception('Classification Error')
                    
                        for p_index in possible_indices:
                            if p_index[0] not in possible_indices_dict:
                                possible_indices_dict[p_index[0]] = []
                            if p_index[1] not in possible_indices_dict[p_index[0]]:
                                possible_indices_dict[p_index[0]].append(p_index[1])
                            if p_index[0] not in possible_indices_keys_alltypes[0]:
                                possible_indices_keys_alltypes[0].append(p_index[0])
                                
                        possible_indices_keys_alltypes[0].sort()
                            
                if target_type in [0, 1]:
                    if (((("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion)) and data_type == "validation dataset") or ((("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and data_type == "test dataset")) and ('cxr_input' in data_info):
                        earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                        possible_indices_keys_alltypes[1]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time<=i])    
                        possible_indices_keys_alltypes[2]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time>i])    
                    elif ('train-missing' in args.modality_inclusion and data_type == "validation dataset") or ('test-missing' in args.modality_inclusion and data_type == "test dataset"):
                        possible_indices_keys_alltypes[2]= list(possible_indices_keys_alltypes[0])
                
                if ("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion and target_type != 2 and data_type == "validation dataset") or ("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion and target_type != 2 and data_type == "test dataset"):
                    if not possible_indices_keys_alltypes[1]: 
                        continue
                
                if target == 1 or target_type == 2:
                    if target == 1:
                        all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)]
                        possible_indices_keys_alltypes[3] = [item for item in all_indices_keys if item not in possible_indices_keys_alltypes[0]]

                    if (((("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion)) and data_type == "validation dataset") or ((("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and data_type == "test dataset")) and ('cxr_input' in data_info):
                        earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                        possible_indices_keys_alltypes[4]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time<=i])
                        possible_indices_keys_alltypes[5]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time>i])    
                    elif ('train-missing' in args.modality_inclusion and data_type == "validation dataset") or ('test-missing' in args.modality_inclusion and data_type == "test dataset"):
                        possible_indices_keys_alltypes[5]= list(possible_indices_keys_alltypes[3])    
                
                possibleWinSizes = data_info['possibleWinSizes']
                possible_indices_keys_alltypes = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices_keys_alltypes])
                
                if isListEmpty(possible_indices_keys_alltypes):
                    continue
                        
                for pidx, possible_indices_keys_type in enumerate(possible_indices_keys_alltypes):
                    if len(possible_indices_keys_type) == 0:
                        continue
                    if pidx < 3:
                        if len(possible_indices_keys_type) >= args.PatPosSampleN:
                            possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatPosSampleN)
                    else:        
                        if len(possible_indices_keys_type) >= args.PatNegSampleN:
                            possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatNegSampleN)

                if data_type == "test dataset":
                    patDict[(pat_id, chid)] = possible_indices_keys_alltypes, possible_indices_dict, target, possibleWinSizes, target_type
            
            patient_list.append(target_type)
            # if target_type == 1 and len(possible_indices_keys_alltypes[3]) > 0 and target_type != 2:
            #     patient_list.append(2)
            ######################################################
            if (('test-full' in args.modality_inclusion and data_type == "validation dataset") or ('test-full' in args.modality_inclusion and data_type == "test dataset")) and "img1" not in args.fullmodal_definition: # (Case1: full_modal with img1 not in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [0, 3]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    for selected_key in possible_indices_keys:
                        if keylist_type == 0:               
                            _data_list.append([pkl_path, [selected_key], possible_indices_dict, possibleWinSizes, target])
                            _type_list.append(target_type)
                            
                        else:                    
                            if target == 1:
                                _data_list.append([pkl_path, [selected_key], {}, possibleWinSizes, 0])
                                _type_list.append(2)
                                
            elif ('test-full' in args.modality_inclusion and "img1" in args.fullmodal_definition and data_type == "validation dataset") or ('test-full' in args.modality_inclusion and "img1" in args.fullmodal_definition and data_type == "test dataset"): # (Case2: full_modal with img1 in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 4]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    for selected_key in possible_indices_keys:
                        if keylist_type == 0:               
                            _data_list.append([pkl_path, [selected_key], possible_indices_dict, possibleWinSizes, target])
                            _type_list.append(target_type)
                            
                        else:                    
                            if target == 1:
                                _data_list.append([pkl_path, [selected_key], {}, possibleWinSizes, 0])
                                _type_list.append(2)
                            
            else: # (Case3: missing modal)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 2, 4, 5]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type < 2:               
                        for selected_key in possible_indices_keys:
                            _data_list.append([pkl_path, [selected_key], possible_indices_dict, possibleWinSizes, target])

                            if keylist_type == 0 and target_type == 1 and "txt1" in file_name:
                                _type_list.append(0)
                            elif keylist_type == 0 and target_type == 0 and "txt1" in file_name:
                                _type_list.append(2)
                            elif keylist_type == 0 and target_type == 1 and "txt1" not in file_name:
                                _type_list.append(3)
                            elif keylist_type == 0 and target_type == 0 and "txt1" not in file_name:
                                _type_list.append(5)
                            elif keylist_type == 1 and target_type == 1 and "txt1" in file_name:
                                _type_list.append(6)
                            elif keylist_type == 1 and target_type == 0 and "txt1" in file_name:
                                _type_list.append(8)
                            elif keylist_type == 1 and target_type == 1 and "txt1" not in file_name:
                                _type_list.append(9)
                            elif keylist_type == 1 and target_type == 0 and "txt1" not in file_name:
                                _type_list.append(11)
                            else:
                                print("Missing modal error with keylist_type < 2")
                                exit(1)
                    else:                    
                        for selected_key in possible_indices_keys:
                            _data_list.append([pkl_path, [selected_key], {}, possibleWinSizes, 0])
                            if keylist_type == 2 and "txt1" in file_name:
                                _type_list.append(1)
                            elif keylist_type == 2 and "txt1" not in file_name:
                                _type_list.append(4)
                            elif keylist_type == 3 and "txt1" in file_name:
                                _type_list.append(7)
                            elif keylist_type == 3 and "txt1" not in file_name:
                                _type_list.append(10)
                            else:
                                print("Missing modal error with keylist_type >= 2")
                                exit(1)
        
        if ('test-full' in args.modality_inclusion):
            ### class 2 방식
            _type_list = [class2dict_full[i] if i in class2dict_full else i for i in _type_list]
        else:
            ### class 2 방식
            _type_list = [class2dict_missing[i] if i in class2dict_missing else i for i in _type_list]    
            
        for idx, sample in enumerate(_data_list):
            pkl_pth, p_key, p_dict, possibleWinSizes, t = sample
            p_key = p_key[0]
            t_type = _type_list[idx]
            win_key_name = "_".join(pkl_pth.split("/")[-1].split("_")[:2]) + f"_{str(p_key)}"
            if win_key_name in winDict:     
                win_size = winDict[win_key_name]
            else:
                win_size = random.choice(possibleWinSizes[p_key])
                winDict[win_key_name] = win_size

            if p_key is not None:
                self._data_list.append([pkl_pth, [p_key], p_dict, win_size, t]) # pkl_path, possible_indices_keys, labels_by_dict, win_size, target
                self._type_list.append(t_type)
                
                if p_key in p_dict:
                    positive_tpoints += 1
                else:
                    negative_tpoints += 1
                        
        if data_type == "test dataset" and load_flag == False:
            with open(test_index_file, 'wb') as f:
                pickle.dump(patDict, f, pickle.HIGHEST_PROTOCOL)
            with open(test_winsize_file, 'wb') as f:
                pickle.dump(winDict, f, pickle.HIGHEST_PROTOCOL)
        
        self.feature_means = list(data_info['mean'])

        print("No Dataset Error 1st: ", len(_type_list) == len(_data_list))
        print("No Dataset Error 2nd: ", len(self._type_list) == len(self._data_list))
        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
        if ('test-full' in args.modality_inclusion):
            print("Number of patient positive samples for training: {}".format(str(patient_list.count(1))))
            print("Number of patient negative samples for training: {}".format(str(patient_list.count(2))))
            print("Number of non-patient negative samples for training: {}".format(str(patient_list.count(0))))        
            print("Number of total negative samples for training: {}".format(str(patient_list.count(0) + patient_list.count(2))))        
        else: # missing modality
            print("[1]. Number of patients: ", patient_list.count(1))
            print("1. Number of patient positive samples for training", self._type_list.count(0)+self._type_list.count(1)+self._type_list.count(2)+self._type_list.count(3))
            print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
            print(" 1-2. wimg-w/otxt_pp: ", self._type_list.count(1))
            print(" 1-3. w/oimg-wtxt_pp: ", self._type_list.count(2))
            print(" 1-4. w/oimg-w/otxt_pp: ", self._type_list.count(3))
            print("[2]. Number of patients with negative signs: ", patient_list.count(2))
            # print("2. Number of patient negative samples list for training", self._type_list.count(8)+self._type_list.count(9)+self._type_list.count(10)+self._type_list.count(11))
            # print(" 2-1. wimg-wtxt-pn: ", self._type_list.count(8))
            # print(" 2-2. wimg-w/otxt-pn: ", self._type_list.count(9))
            # print(" 2-3. w/oimg-wtxt-pn: ", self._type_list.count(10))
            # print(" 2-4. w/oimg-w/otxt-pn: ", self._type_list.count(11))
            print("[3]. Number of non-patients: ", patient_list.count(0))
            print("3. Number of non-patient negative samples for training", self._type_list.count(4)+self._type_list.count(5)+self._type_list.count(6)+self._type_list.count(7))
            print(" 3-1. wimg-wtxt-nn: ", self._type_list.count(4))
            print(" 3-2. wimg-w/otxt-nn: ", self._type_list.count(5))
            print(" 3-3. w/oimg-wtxt-nn: ", self._type_list.count(6))
            print(" 3-4. w/oimg-w/otxt-nn: ", self._type_list.count(7))
        print("########## Detail Data Info ##########")
        print("Positive time-points: ", positive_tpoints)
        print("Negative time-points: ", negative_tpoints)
        
        self.feature_mins = self.train_min
        self.feature_maxs = self.train_max
        
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        pkl_path, possible_indices_keys, labels_by_dict, win_size, target = self._data_list[index]
        type_list = self._type_list[index]
        
        file_name = pkl_path.split("/")[-1]
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

        windowIndex = self.window_size - 1
        selectedKey = possible_indices_keys[0]
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        f_indices = False
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                    
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, self.window_size, self.vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = self.neg_multi_target
        else:
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // self.intv_len)
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)
        multi_target = torch.tensor(multi_target)

        missing = [False]   # Missing modality list: [vital/lab, img, txt]
        
        if (("img" in args.input_types and "img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion and type_list in [0,2,3,5] and "img" in args.input_types)) and ('cxr_input' in data_pkl):
            cxr_li = [i for i in data_pkl['cxr_input'] if i[0] <= selectedKey] 
            if not cxr_li and ('test-full' in args.modality_inclusion): 
                print("collate cxr error")
                exit(1)
            elif not cxr_li and ('test-missing' in args.modality_inclusion): 
                img = torch.zeros(self.image_size).unsqueeze(0)
                missing.append(True)
            else:
                cxr_path = sorted(cxr_li)[-1][1]
                image = Image.open(self.image_data_path + cxr_path)
                image = F_t.equalize(image)
                img = self.transform(image)
                missing.append(False)
        else:
            img = torch.zeros(self.image_size).unsqueeze(0)
            missing.append(True)
        
        if (("txt" in args.input_types and "txt1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion and "txt" in args.input_types)) and ("txt1" in file_name):
            tokens = self.txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
            if len(tokens) == 0:
                tokens = torch.zeros(self.token_max_length)
                textLength = 0
                missing.append(True)
            else:
                textLength = len(tokens)
                # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
                # EX) 2 {Sentence Tokens} {1 Padding} 3
                # Add Beginnning of Sentence Token
                tokens.insert(0, 2)
                tokens = torch.Tensor(clinical_note_transform(tokens))
                tokens[tokens==1] = 0
                missing.append(False)
        else:    
            tokens = torch.zeros(self.token_max_length)
            textLength = 0
            missing.append(True)
        missing = torch.Tensor(missing)
        return final_seqs, static_inputs, multi_target, inputLength, img, tokens, textLength, missing, f_indices

class Multiple_Outbreaks_Training_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Training Dataset for Multiple_Outbreaks_Dataset Prediction...")
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        patient_list = []
        
        self.feature_mins = None
        self.feature_maxs = None
        self.transform = None
        self.txtDict = None
        
        lengths = []
        positive_tpoints = 0
        negative_tpoints = 0
        
        self.vslt_len = len(args.vitalsign_labtest)
        self.neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.intv_len = int(args.prediction_range // 12)
        self.window_size = args.window_size
        self.image_size = [args.image_size, args.image_size]
        self.input_types = args.input_types
        self.image_data_path = args.image_data_path
        self.token_max_length = args.bert_token_max_length
        self.txtDict = txtDictLoad("train")
        
        class2dict_missing = {3:1, 6:2, 9:3, 2:4, 8:6, 11:7, 1:4, 4:5, 7:6, 10:7}
        class2dict_full = {2:0}

        lengths = []
        tmpTasks = ['vasso', 'intubation', 'cpr']
        tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
        tmptimes = ['vasso_time', 'intubation_time', 'cpr_time']
        taskIndex = tmpTasks.index(args.output_type)
        # -> ex) if args.output_type = "vasso", data_pkl['vasso_time'] can be written as data_pkl[tmpInputs[taskIndex]]

        # real-time x-ray image transform function
        if ("img" in self.input_types or 'train-missing' in args.modality_inclusion):
            self.transform = xray_image_transform_train()
                
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            if "cxr_input" in data_info:
                new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
                if len(new_cxr_inputs) > 0:
                    data_info["cxr_input"] = new_cxr_inputs
                else:
                    del data_info['cxr_input']
                    file_name = file_name.replace("_img1", "_img0")
                
            if 'train-full' in args.modality_inclusion:
                if args.fullmodal_definition not in file_name:
                    continue
                if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                    continue
                if "txt1" in args.fullmodal_definition:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        continue
            else: # missing modality
                if "txt1" in file_name:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        file_name = file_name.replace("_txt1_", "_txt0_")
                if ("cxr_input" not in data_info and "img1" in file_name):
                    file_name = file_name.replace("_img1", "_img0")
                
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue            
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                continue
            
            ##### get possibles indices and max lengths here #####
            # If there are no positive cases in indices below args.min_inputlen, we change the case to negative.
            # If outbreak_time is beyond the prediction range of the given data or happened too early, change to 0 target
            outbreak_times = data_info[tmptimes[taskIndex]]
            if outbreak_times is not None and len(outbreak_times) != 0:
                outbreak_times = sorted(outbreak_times)
                if isinstance(outbreak_times[0], tuple):
                    outbreak_times = list([i for i in outbreak_times if (i[0] >= args.min_inputlen) and (i[0] <= sequenceLength + args.prediction_range - 1)])
                else:
                    outbreak_times = list([i for i in outbreak_times if (i >= args.min_inputlen) and (i <= sequenceLength + args.prediction_range - 1)])
                
                if len(outbreak_times) == 0:
                    target = 0
                else:
                    target = 1
            else:
                target = 0

            # If target variable negative, get random subsequence of length 3 ~ args.window_size
            possible_indices_dict = {}            
            possible_indices_keys_alltypes = [[] for _ in range(6)]
            if target == 0:
                target_type = 0
                possible_indices_keys_alltypes[0] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
                
            else:
                # For within-n-hours task, the data must be within prediction_range of outbreak times
                # range 2~3 means 2 < x =< 3
                dup_counts = []
                possible_indices_list = []
                target_type = 1
                for idx_outbreak, outbreak_time in enumerate(outbreak_times):
                    if isinstance(outbreak_times[0], tuple):
                        outbreak_time = math.ceil(outbreak_time[0])
                    else:
                        outbreak_time = math.ceil(outbreak_time)
                    if outbreak_time in dup_counts:
                        continue
                    else:
                        dup_counts.append(outbreak_time)
                    possible_indices = [(outbreak_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_time >= args.min_inputlen+i-1) and (outbreak_time - i < sequenceLength)] or None
                    
                    if possible_indices is None:
                        print("SeqLength : " + str(sequenceLength))
                        print(data_info)
                        raise Exception('Classification Error')
                    possible_indices_list.append(possible_indices)
                    # print(f"{str(idx_outbreak)}: ", possible_indices)
                    
                # p_indices: [(32, [0, 1]), (31, [1, 2]), (30, [2, 3]), (29, [3, 4]), (28, [4, 5]), (27, [5, 6]), (26, [6, 7]), (25, [7, 8]), (24, [8, 9]), (23, [9, 10]), (22, [10, 11]), (21, [11, 12])]
                for p_indices in possible_indices_list: 
                    # p_index: (32, [0, 1])
                    for p_index in p_indices:
                        if p_index[0] not in possible_indices_dict:
                            possible_indices_dict[p_index[0]] = []
                        if p_index[1] not in possible_indices_dict[p_index[0]]:
                            possible_indices_dict[p_index[0]].append(p_index[1])
                        if p_index[0] not in possible_indices_keys_alltypes[0]:
                            possible_indices_keys_alltypes[0].append(p_index[0])
                possible_indices_keys_alltypes[0].sort()
            if (("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                possible_indices_keys_alltypes[1]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time<=i])    
                possible_indices_keys_alltypes[2]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time>i])    
            elif ('train-missing' in args.modality_inclusion):
                possible_indices_keys_alltypes[2]= list(possible_indices_keys_alltypes[0])
                
            if ("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion):
                if not possible_indices_keys_alltypes[1]: 
                    continue
            
            patient_list.append(target_type)
                
            if target == 1:
                all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)]
                possible_indices_keys_alltypes[3] = [item for item in all_indices_keys if item not in possible_indices_keys_alltypes[0]]
                
                if len(possible_indices_keys_alltypes[3]) > 0:
                    if ('train-full' in args.modality_inclusion and "img1" in args.fullmodal_definition) or ('train-missing' in args.modality_inclusion) and ('cxr_input' in data_info):
                        possible_indices_keys_alltypes[4]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time<=i])
                        possible_indices_keys_alltypes[5]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time>i])    
                    elif ('train-missing' in args.modality_inclusion):
                        possible_indices_keys_alltypes[5]= list(possible_indices_keys_alltypes[3])  
                        
                    if len(possible_indices_keys_alltypes[0]) == 0:
                        patient_list.append(2)
            # if file_name == "10329745_20173197_txt1_img1.pkl":
            #     print(possible_indices_keys_alltypes)
            #     print(target)
            #     print(target_type)
            #     exit(1)
            ######################################################
            possibleWinSizes = data_info['possibleWinSizes']
            # possible_indices_keys = [key for key in possible_indices_keys if key in possibleWinSizes]
            possible_indices_keys_alltypes = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices_keys_alltypes])
            
            if isListEmpty(possible_indices_keys_alltypes):
                continue

            ######################################################
            if ('train-full' in args.modality_inclusion and "img1" not in args.fullmodal_definition): # (Case1: full_modal with img1 not in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [0, 3]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type == 0:               
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:# possible_indices_keys가 빈 리스트라면 실행 안됨
                            self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, possibleWinSizes, target])
                            self._type_list.append(target_type)
                            
                            possible_tpoints = [True if i in possible_indices_dict else False for i in possible_indices_keys]
                            positive_tpoints += possible_tpoints.count(True)
                            negative_tpoints += possible_tpoints.count(False)
                    else:                    
                        if target == 1 and len(possible_indices_keys) > 0 and possible_indices_keys is not None:
                            self._data_list.append([pkl_path, possible_indices_keys, {}, possibleWinSizes, 0])
                            self._type_list.append(2)
                            negative_tpoints += len(possible_indices_keys)
                                
            elif ('train-full' in args.modality_inclusion and "img1" in args.fullmodal_definition): # (Case2: full_modal with img1 in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 4]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type == 0:               
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:# possible_indices_keys가 빈 리스트라면 실행 안됨
                            self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, possibleWinSizes, target])
                            self._type_list.append(target_type)
                            
                            possible_tpoints = [True if i in possible_indices_dict else False for i in possible_indices_keys]
                            positive_tpoints += possible_tpoints.count(True)
                            negative_tpoints += possible_tpoints.count(False)
                    else:                    
                        if target == 1 and len(possible_indices_keys) > 0 and possible_indices_keys is not None:
                            self._data_list.append([pkl_path, possible_indices_keys, {}, possibleWinSizes, 0])
                            self._type_list.append(2)
                            negative_tpoints += len(possible_indices_keys)
                            
            else: # (Case3: missing modal)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 2, 4, 5]])
                
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type < 2:               
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:# possible_indices_keys가 빈 리스트라면 실행 안됨
                            self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, possibleWinSizes, target])
                            if keylist_type == 0 and target_type == 1 and "txt1" in file_name:
                                self._type_list.append(0)
                            elif keylist_type == 0 and target_type == 0 and "txt1" in file_name:
                                self._type_list.append(2)
                            elif keylist_type == 0 and target_type == 1 and "txt1" not in file_name:
                                self._type_list.append(3)
                            elif keylist_type == 0 and target_type == 0 and "txt1" not in file_name:
                                self._type_list.append(5)
                            elif keylist_type == 1 and target_type == 1 and "txt1" in file_name:
                                self._type_list.append(6)
                            elif keylist_type == 1 and target_type == 0 and "txt1" in file_name:
                                self._type_list.append(8)
                            elif keylist_type == 1 and target_type == 1 and "txt1" not in file_name:
                                self._type_list.append(9)
                            elif keylist_type == 1 and target_type == 0 and "txt1" not in file_name:
                                self._type_list.append(11)
                            else:
                                print("Missing modal error with keylist_type < 2")
                                exit(1)
                            possible_tpoints = [True if i in possible_indices_dict else False for i in possible_indices_keys]
                            positive_tpoints += possible_tpoints.count(True)
                            negative_tpoints += possible_tpoints.count(False)
                    else:                    
                        if len(possible_indices_keys) > 0 and possible_indices_keys is not None:
                            self._data_list.append([pkl_path, possible_indices_keys, {}, possibleWinSizes, 0])
                            if keylist_type == 2 and "txt1" in file_name:
                                self._type_list.append(1)
                            elif keylist_type == 2 and "txt1" not in file_name:
                                self._type_list.append(4)
                            elif keylist_type == 3 and "txt1" in file_name:
                                self._type_list.append(7)
                            elif keylist_type == 3 and "txt1" not in file_name:
                                self._type_list.append(10)
                            else:
                                print("Missing modal error with keylist_type >= 2")
                                exit(1)
                                # print("### 2 ###")
                                # print("keylist_type: ", keylist_type)
                                # print("file_name: ", file_name)

                            negative_tpoints += len(possible_indices_keys)
            
            ######################################################
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)
                
        if ('train-full' in args.modality_inclusion):
            ### class 2 방식
            self._type_list = [class2dict_full[i] if i in class2dict_full else i for i in self._type_list]
        else:
            ### class 2 방식
            self._type_list = [class2dict_missing[i] if i in class2dict_missing else i for i in self._type_list]
            
        self.feature_means = list(data_info['mean'])
        print("No Dataset Error: ", len(self._type_list) == len(self._data_list))
        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
        if ('train-full' in args.modality_inclusion):
            print("Number of patient positive samples list for training: {}".format(str(patient_list.count(1))))
            print("Number of patient negative samples list for training: {}".format(str(patient_list.count(2))))
            print("Number of non-patient negative samples list for training: {}".format(str(patient_list.count(0))))        
            print("Number of total negative samples list for training: {}".format(str(patient_list.count(0) + patient_list.count(2))))        
        else: # missing modality
            print("[1]. Number of patients: ", patient_list.count(1))
            print("1. Number of patient positive samples lists for training", self._type_list.count(0)+self._type_list.count(1)+self._type_list.count(2)+self._type_list.count(3))
            print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
            print(" 1-2. wimg-w/otxt_pp: ", self._type_list.count(1))
            print(" 1-3. w/oimg-wtxt_pp: ", self._type_list.count(2))
            print(" 1-4. w/oimg-w/otxt_pp: ", self._type_list.count(3))
            print("[2]. Number of patients with negative signs: ", patient_list.count(2))
            # print("2. Number of patient negative samples list for training", self._type_list.count(8)+self._type_list.count(9)+self._type_list.count(10)+self._type_list.count(11))
            # print(" 2-1. wimg-wtxt-pn: ", self._type_list.count(8))
            # print(" 2-2. wimg-w/otxt-pn: ", self._type_list.count(9))
            # print(" 2-3. w/oimg-wtxt-pn: ", self._type_list.count(10))
            # print(" 2-4. w/oimg-w/otxt-pn: ", self._type_list.count(11))
            print("[3]. Number of non-patients: ", patient_list.count(0))
            print("3. Number of non-patient negative samples lists for training", self._type_list.count(4)+self._type_list.count(5)+self._type_list.count(6)+self._type_list.count(7))
            print(" 3-1. wimg-wtxt-nn: ", self._type_list.count(4))
            print(" 3-2. wimg-w/otxt-nn: ", self._type_list.count(5))
            print(" 3-3. w/oimg-wtxt-nn: ", self._type_list.count(6))
            print(" 3-4. w/oimg-w/otxt-nn: ", self._type_list.count(7))
            # print("[1]. Number of patients: ", patient_list.count(1))
            # print("1. Number of patient positive samples list for training", self._type_list.count(0)+self._type_list.count(3)+self._type_list.count(6)+self._type_list.count(9))
            # print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
            # print(" 1-2. wimg-w/otxt_pp: ", self._type_list.count(3))
            # print(" 1-3. w/oimg-wtxt_pp: ", self._type_list.count(6))
            # print(" 1-4. w/oimg-w/otxt_pp: ", self._type_list.count(9))
            # print("[2]. Number of patients with negative signs: ", patient_list.count(2))
            # # print("2. Number of patient negative samples list for training", self._type_list.count(1)+self._type_list.count(4)+self._type_list.count(7)+self._type_list.count(10))
            # # print(" 2-1. wimg-wtxt-pn: ", self._type_list.count(1))
            # # print(" 2-2. wimg-w/otxt-pn: ", self._type_list.count(4))
            # # print(" 2-3. w/oimg-wtxt-pn: ", self._type_list.count(7))
            # # print(" 2-4. w/oimg-w/otxt-pn: ", self._type_list.count(10))
            # print("[3]. Number of non-patients: ", patient_list.count(0))
            # print("3. Number of non-patient negative samples list for training", self._type_list.count(2)+self._type_list.count(5)+self._type_list.count(8)+self._type_list.count(11))
            # print(" 3-1. wimg-wtxt-nn: ", self._type_list.count(2))
            # print(" 3-2. wimg-w/otxt-nn: ", self._type_list.count(5))
            # print(" 3-3. w/oimg-wtxt-nn: ", self._type_list.count(8))
            # print(" 3-4. w/oimg-w/otxt-nn: ", self._type_list.count(11))
        print("########## Detail Data Info ##########")
        print("Positive time-points: ", positive_tpoints)
        print("Negative time-points: ", negative_tpoints)

        args.feature_mins = self.train_min
        args.feature_maxs = self.train_max

        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        pkl_path, possible_indices_keys, labels_by_dict, win_sizes, target = self._data_list[index]
        type_list = self._type_list[index]
        
        file_name = pkl_path.split("/")[-1]
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

        windowIndex = self.window_size - 1
        
        # if 'train-full' in args.modality_inclusion:
        #     earliest_img_time = min([j[0] for j in data_pkl['cxr_input']])
        #     possible_indices_keys= list([i for i in possible_indices_keys if earliest_img_time<=i])
        
        selectedKey = random.choice(possible_indices_keys)
        randLength = random.choice(win_sizes[selectedKey])
        
        if args.auxiliary_loss_input is None:
            dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, randLength, windowIndex, data_pkl)
            f_indices = False
        else:
            dataSequence, maskSequence, deltaSequence, inputLength, f_indices = sequenceGenerator_pretrain(args, selectedKey, randLength, windowIndex+12, data_pkl)
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-randLength) -1
                
                if (i <= selectedKey) and (i > selectedKey-randLength):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                    
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, self.window_size, self.vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = self.neg_multi_target
        else:
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // self.intv_len)
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)
        multi_target = torch.tensor(multi_target)
        
        missing = [False]   # Missing modality list: [vital/lab, img, txt]
        
        if (("img" in args.input_types and "img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion and type_list in [0,2,3,5] and "img" in args.input_types)) and ('cxr_input' in data_pkl):
            cxr_li = [i for i in data_pkl['cxr_input'] if i[0] <= selectedKey] 
            if not cxr_li and ('train-full' in args.modality_inclusion): 
                print("collate cxr error")
                exit(1)
            elif not cxr_li and ('train-missing' in args.modality_inclusion): 
                img = torch.zeros(self.image_size).unsqueeze(0)
                missing.append(True)
            else:
                cxr_path = sorted(cxr_li)[-1][1]
                image = Image.open(self.image_data_path + cxr_path)
                image = F_t.equalize(image)
                img = self.transform(image)
                missing.append(False)
        else:
            img = torch.zeros(self.image_size).unsqueeze(0)
            missing.append(True)
        
        if (("txt" in args.input_types and "txt1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion and "txt" in args.input_types)) and ("txt1" in file_name):
            tokens = self.txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
            if len(tokens) == 0:
                tokens = torch.zeros(self.token_max_length)
                textLength = 0
                missing.append(True)
            else:
                textLength = len(tokens)
                # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
                # EX) 2 {Sentence Tokens} {1 Padding} 3
                # Add Beginnning of Sentence Token
                tokens.insert(0, 2)
                tokens = torch.Tensor(clinical_note_transform(tokens))
                tokens[tokens==1] = 0
                missing.append(False)
        else:    
            tokens = torch.zeros(self.token_max_length)
            textLength = 0
            missing.append(True)
        missing = torch.Tensor(missing)  
        return final_seqs, static_inputs, multi_target, inputLength, img, tokens, textLength, missing, f_indices

class Multiple_Outbreaks_Test_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preparing {} Dataset for Multiple_Outbreaks_Dataset Prediction...".format(data_type))

        self._data_list = []
        _data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        _type_list = []
        patient_list = []
        patDict = {}
        winDict = {}
        
        self.feature_mins = None
        self.feature_maxs = None
        self.transform = None
        self.txtDict = None
        
        lengths = []
        positive_tpoints = 0
        negative_tpoints = 0
        
        self.vslt_len = len(args.vitalsign_labtest)
        self.neg_multi_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.intv_len = int(args.prediction_range // 12)
        self.window_size = args.window_size
        self.image_size = [args.image_size, args.image_size]
        self.input_types = args.input_types
        self.image_data_path = args.image_data_path
        self.token_max_length = args.bert_token_max_length
        
        class2dict_missing = {3:1, 6:2, 9:3, 2:4, 8:6, 11:7, 1:4, 4:5, 7:6, 10:7}
        class2dict_full = {2:0}
        
        load_flag = False
        tmpTasks = ['vasso', 'intubation', 'cpr']
        tmpYN = ['vasso_yn', 'intubation_yn', 'cpr_yn']
        tmptimes = ['vasso_time', 'intubation_time', 'cpr_time']
        taskIndex = tmpTasks.index(args.output_type)
        # -> ex) if args.output_type = "vasso", data_pkl['vasso_time'] can be written as data_pkl[tmpInputs[taskIndex]]
        
        test_index_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + ".pkl"
        test_winsize_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + "_winsize.pkl"
        validation_index_file = "./data/testIndexes/valIndexes__" + args.train_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + ".pkl"
        validation_winsize_file = "./data/testIndexes/valIndexes__" + args.train_data_path.split("/")[-2] + "__" + args.modality_inclusion.split("_")[-1] + "__fullmodaldefinition" + str(args.fullmodal_definition) + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + "_winsize.pkl"
        
        if not os.path.exists("./data/testIndexes"):
            os.makedirs('./data/testIndexes', exist_ok=True)
            
        if data_type == "validation dataset":
            self.txtDict = txtDictLoad("train")
        else:
            self.txtDict = txtDictLoad("test")

        if data_type == "test dataset":
            if  os.path.exists(test_index_file) and os.path.exists(test_winsize_file) and data_type == "test dataset":
                # Open the file and add existing entries to dictionary
                print("Index file exists... Loading...")
                load_flag = True
                with open(test_index_file, 'rb') as f:
                    patDict = pickle.load(f)
                with open(test_winsize_file, 'rb') as f:
                    winDict = pickle.load(f)
            else:
                if os.path.exists(test_index_file):
                    os.remove(test_index_file)
                if os.path.exists(test_winsize_file):
                    os.remove(test_winsize_file)
                
        elif data_type == "validation dataset":
            if  os.path.exists(validation_index_file) and os.path.exists(validation_winsize_file) and data_type == "validation dataset":
                # Open the file and add existing entries to dictionary
                print("Index file exists... Loading...")
                load_flag = True
                with open(validation_index_file, 'rb') as f:
                    patDict = pickle.load(f)
                with open(validation_winsize_file, 'rb') as f:
                    winDict = pickle.load(f)
            else:
                print("Index file not exists... Generating...")
                if os.path.exists(validation_index_file):
                    os.remove(validation_index_file)
                if os.path.exists(validation_winsize_file):
                    os.remove(validation_winsize_file)        
                
                multiple_outbreaks_valdataset_maker(args, validation_index_file, validation_winsize_file)
                with open(validation_index_file, 'rb') as f:
                    patDict = pickle.load(f)
                with open(validation_winsize_file, 'rb') as f:
                    winDict = pickle.load(f)
                    
        else:
            raise Exception('Data Type Error')

        # real-time x-ray image transform function
        if "img" in self.input_types:
            if args.image_test_type == "center":
                self.transform = xray_image_transform_center_val()
            elif args.image_test_type == "resize":
                self.transform = xray_image_transform_resize_val()

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            if "cxr_input" in data_info:
                new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
                if len(new_cxr_inputs) > 0:
                    data_info["cxr_input"] = new_cxr_inputs
                else:
                    del data_info['cxr_input']
                    file_name = file_name.replace("_img1", "_img0")
                
            if ('test-full' in args.modality_inclusion and data_type == "validation dataset") or ('test-full' in args.modality_inclusion and data_type == "test dataset"):
                if args.fullmodal_definition not in file_name:
                    continue
                if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                    continue
                if "txt1" in args.fullmodal_definition:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        continue
            else: # missing modality
                if "txt1" in file_name:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        file_name = file_name.replace("_txt1_", "_txt0_")
                if ("cxr_input" not in data_info and "img1" in file_name):
                    file_name = file_name.replace("_img1", "_img0")
            
            pat_id = int(data_info['pat_id'])
            chid = int(data_info['chid'])
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                continue
                
            # Check if the randIndex for the given patient has already been initialized
            if (pat_id, chid) in patDict:
                possible_indices_keys_alltypes, possible_indices_dict, target, possibleWinSizes, target_type = patDict[(pat_id, chid)]
                
                if isListEmpty(possible_indices_keys_alltypes):
                    continue
                
            else:
                if data_type == "validation dataset":
                    continue
                ##### get possibles indices and max lengths here #####
                # If there are no positive cases in indices below args.min_inputlen, we change the case to negative.
                # If outbreak_time is beyond the prediction range of the given data or happened too early, change to 0 target
                outbreak_times = data_info[tmptimes[taskIndex]]
                if outbreak_times is not None and len(outbreak_times) != 0:
                    outbreak_times = sorted(outbreak_times)
                    if isinstance(outbreak_times[0], tuple):
                        outbreak_times = list([i for i in outbreak_times if (i[0] >= args.min_inputlen) and (i[0] <= sequenceLength + args.prediction_range - 1)])
                    else:
                        outbreak_times = list([i for i in outbreak_times if (i >= args.min_inputlen) and (i <= sequenceLength + args.prediction_range - 1)])
                    
                    if len(outbreak_times) == 0:
                        target = 0
                    else:
                        target = 1
                else:
                    target = 0

                possible_indices_dict = {}
                # possible_indices_keys: 0 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pp, nn)
                # possible_indices_keys_with_img: 1 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pp, nn)     Case3: (wimgwtxt_pp: 0, wimgwtxt-nn: 2, wimgw/otxt_pp: 3, wimgw/otxt-nn: 5)
                # possible_indices_keys_without_img: 2 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt_pp: 6, w/oimgwtxt-nn: 8, w/oimgw/otxt_pp: 9, w/oimgw/otxt-nn: 11)
                # pat_neg_indices_keys = 3 for (Case1: full_modal with img1 not in fullmodal_definition)                                    Case1: (pn)
                # pat_neg_indices_keys_with_img = 4 for (Case2: full_modal with img1 in fullmodal_definition) or (Case3: missing modal)     Case2: (pn)         Case3: (wimgwtxt-pn: 1, wimgw/otxt-pn: 4)
                # pat_neg_indices_keys_without_img = 5 for (Case3: missing modal)                                                                               Case3: (w/oimgwtxt-pn: 7, w/oimgw/otxt-pn: 10)
                possible_indices_keys_alltypes = [[] for _ in range(6)]
                
                possibleWinSizes = data_info['possibleWinSizes']
                
                if target == 0:
                    target_type = 0
                    possible_indices_keys_alltypes[0] = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
            
                else:
                    # For within-n-hours task, the data must be within prediction_range of outbreak times
                    # range 2~3 means 2 < x =< 3
                    dup_counts = []
                    possible_indices_list = []
                    target_type = 1
                    for idx_outbreak, outbreak_time in enumerate(outbreak_times):
                        if isinstance(outbreak_times[0], tuple):
                            outbreak_time = math.ceil(outbreak_time[0])
                        else:
                            outbreak_time = math.ceil(outbreak_time)
                        if outbreak_time in dup_counts:
                            continue
                        else:
                            dup_counts.append(outbreak_time)
                        possible_indices = [(outbreak_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_time >= args.min_inputlen+i-1) and (outbreak_time - i < sequenceLength)] or None
                        
                        if possible_indices is None:
                            print("SeqLength : " + str(sequenceLength))
                            print(data_info)
                            raise Exception('Classification Error')
                        possible_indices_list.append(possible_indices)
                        # print(f"{str(idx_outbreak)}: ", possible_indices)
                        
                    # p_indices: [(32, [0, 1]), (31, [1, 2]), (30, [2, 3]), (29, [3, 4]), (28, [4, 5]), (27, [5, 6]), (26, [6, 7]), (25, [7, 8]), (24, [8, 9]), (23, [9, 10]), (22, [10, 11]), (21, [11, 12])]
                    for p_indices in possible_indices_list: 
                        # p_index: (32, [0, 1])
                        for p_index in p_indices:
                            if p_index[0] not in possible_indices_dict:
                                possible_indices_dict[p_index[0]] = []
                            if p_index[1] not in possible_indices_dict[p_index[0]]:
                                possible_indices_dict[p_index[0]].append(p_index[1])
                            if p_index[0] not in possible_indices_keys_alltypes[0]:
                                possible_indices_keys_alltypes[0].append(p_index[0])
                    possible_indices_keys_alltypes[0].sort()
                
                if (("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                    earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                    possible_indices_keys_alltypes[1]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time<=i])    
                    possible_indices_keys_alltypes[2]= list([i for i in possible_indices_keys_alltypes[0] if earliest_img_time>i])    
                elif ('test-missing' in args.modality_inclusion):
                    possible_indices_keys_alltypes[2]= list(possible_indices_keys_alltypes[0])
                    
                if ("img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion):
                    if not possible_indices_keys_alltypes[1]: 
                        continue
                    
                if target == 1:
                    all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)]
                    possible_indices_keys_alltypes[3] = [item for item in all_indices_keys if item not in possible_indices_keys_alltypes[0]]

                    if len(possible_indices_keys_alltypes[3]) > 0:
                        if ('test-full' in args.modality_inclusion and "img1" in args.fullmodal_definition) or ('test-missing' in args.modality_inclusion) and ('cxr_input' in data_info):
                            possible_indices_keys_alltypes[4]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time<=i])
                            possible_indices_keys_alltypes[5]= list([i for i in possible_indices_keys_alltypes[3] if earliest_img_time>i])    
                        elif ('test-missing' in args.modality_inclusion):
                            possible_indices_keys_alltypes[5]= list(possible_indices_keys_alltypes[3])  
                
                possibleWinSizes = data_info['possibleWinSizes']
                possible_indices_keys_alltypes = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices_keys_alltypes])
                
                if isListEmpty(possible_indices_keys_alltypes):
                    continue
                        
                for pidx, possible_indices_keys_type in enumerate(possible_indices_keys_alltypes):
                    if len(possible_indices_keys_type) == 0:
                        continue
                    if pidx < 3:
                        if len(possible_indices_keys_type) >= args.PatPosSampleN:
                            possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatPosSampleN)
                    else:        
                        if len(possible_indices_keys_type) >= args.PatNegSampleN:
                            possible_indices_keys_alltypes[pidx] = random.sample(possible_indices_keys_type, args.PatNegSampleN)

                if data_type == "test dataset":
                    patDict[(pat_id, chid)] = possible_indices_keys_alltypes, possible_indices_dict, target, possibleWinSizes, target_type
            
            patient_list.append(target_type)
            if target_type == 1 and len(possible_indices_keys_alltypes[3]) > 0 and len(possible_indices_keys_alltypes[0]) == 0:
                patient_list.append(2)
            
            ######################################################
            if (('test-full' in args.modality_inclusion and data_type == "validation dataset") or ('test-full' in args.modality_inclusion and data_type == "test dataset")) and "img1" not in args.fullmodal_definition: # (Case1: full_modal with img1 not in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [0, 3]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    for selected_key in possible_indices_keys:
                        if keylist_type == 0:               
                            _data_list.append([pkl_path, [selected_key], possible_indices_dict, possibleWinSizes, target])
                            _type_list.append(target_type)
                            
                        else:                    
                            if target == 1:
                                _data_list.append([pkl_path, [selected_key], {}, possibleWinSizes, 0])
                                _type_list.append(2)
                                
            elif ('test-full' in args.modality_inclusion and "img1" in args.fullmodal_definition and data_type == "validation dataset") or ('test-full' in args.modality_inclusion and "img1" in args.fullmodal_definition and data_type == "test dataset"): # (Case2: full_modal with img1 in fullmodal_definition)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 4]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    for selected_key in possible_indices_keys:
                        if keylist_type == 0:               
                            _data_list.append([pkl_path, [selected_key], possible_indices_dict, possibleWinSizes, target])
                            _type_list.append(target_type)
                            
                        else:                    
                            if target == 1:
                                _data_list.append([pkl_path, [selected_key], {}, possibleWinSizes, 0])
                                _type_list.append(2)
                            
            else: # (Case3: missing modal)
                possible_indices_keys_alltypes = list([i for idx, i in enumerate(possible_indices_keys_alltypes) if idx in [1, 2, 4, 5]])
                for keylist_type, possible_indices_keys in enumerate(possible_indices_keys_alltypes):
                    if keylist_type < 2:               
                        for selected_key in possible_indices_keys:
                            _data_list.append([pkl_path, [selected_key], possible_indices_dict, possibleWinSizes, target])

                            if keylist_type == 0 and target_type == 1 and "txt1" in file_name:
                                _type_list.append(0)
                            elif keylist_type == 0 and target_type == 0 and "txt1" in file_name:
                                _type_list.append(2)
                            elif keylist_type == 0 and target_type == 1 and "txt1" not in file_name:
                                _type_list.append(3)
                            elif keylist_type == 0 and target_type == 0 and "txt1" not in file_name:
                                _type_list.append(5)
                            elif keylist_type == 1 and target_type == 1 and "txt1" in file_name:
                                _type_list.append(6)
                            elif keylist_type == 1 and target_type == 0 and "txt1" in file_name:
                                _type_list.append(8)
                            elif keylist_type == 1 and target_type == 1 and "txt1" not in file_name:
                                _type_list.append(9)
                            elif keylist_type == 1 and target_type == 0 and "txt1" not in file_name:
                                _type_list.append(11)
                            else:
                                print("Missing modal error with keylist_type < 2")
                                exit(1)
                    else:                    
                        for selected_key in possible_indices_keys:
                            _data_list.append([pkl_path, [selected_key], {}, possibleWinSizes, 0])
                            if keylist_type == 2 and "txt1" in file_name:
                                _type_list.append(1)
                            elif keylist_type == 2 and "txt1" not in file_name:
                                _type_list.append(4)
                            elif keylist_type == 3 and "txt1" in file_name:
                                _type_list.append(7)
                            elif keylist_type == 3 and "txt1" not in file_name:
                                _type_list.append(10)
                            else:
                                print("Missing modal error with keylist_type >= 2")
                                exit(1)
        
        if ('test-full' in args.modality_inclusion):
            ### class 2 방식
            _type_list = [class2dict_full[i] if i in class2dict_full else i for i in _type_list]
        else:
            ### class 2 방식
            _type_list = [class2dict_missing[i] if i in class2dict_missing else i for i in _type_list]    
            
        for idx, sample in enumerate(_data_list):
            pkl_pth, p_key, p_dict, possibleWinSizes, t = sample
            p_key = p_key[0]
            t_type = _type_list[idx]
            win_key_name = "_".join(pkl_pth.split("/")[-1].split("_")[:2]) + f"_{str(p_key)}"
            if win_key_name in winDict:     
                win_size = winDict[win_key_name]
            else:
                win_size = random.choice(possibleWinSizes[p_key])
                winDict[win_key_name] = win_size

            if p_key is not None:
                self._data_list.append([pkl_pth, [p_key], p_dict, win_size, t]) # pkl_path, possible_indices_keys, labels_by_dict, win_size, target
                self._type_list.append(t_type)
                
                if p_key in p_dict:
                    positive_tpoints += 1
                else:
                    negative_tpoints += 1
                        
        if data_type == "test dataset" and load_flag == False:
            with open(test_index_file, 'wb') as f:
                pickle.dump(patDict, f, pickle.HIGHEST_PROTOCOL)
            with open(test_winsize_file, 'wb') as f:
                pickle.dump(winDict, f, pickle.HIGHEST_PROTOCOL)
        
        self.feature_means = list(data_info['mean'])

        print("No Dataset Error 1st: ", len(_type_list) == len(_data_list))
        print("No Dataset Error 2nd: ", len(self._type_list) == len(self._data_list))
        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
        if ('test-full' in args.modality_inclusion):
            print("Number of patient positive samples for training: {}".format(str(patient_list.count(1))))
            print("Number of patient negative samples for training: {}".format(str(patient_list.count(2))))
            print("Number of non-patient negative samples for training: {}".format(str(patient_list.count(0))))        
            print("Number of total negative samples for training: {}".format(str(patient_list.count(0) + patient_list.count(2))))        
        else: # missing modality
            print("[1]. Number of patients: ", patient_list.count(1))
            print("1. Number of patient positive samples for training", self._type_list.count(0)+self._type_list.count(1)+self._type_list.count(2)+self._type_list.count(3))
            print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
            print(" 1-2. wimg-w/otxt_pp: ", self._type_list.count(1))
            print(" 1-3. w/oimg-wtxt_pp: ", self._type_list.count(2))
            print(" 1-4. w/oimg-w/otxt_pp: ", self._type_list.count(3))
            print("[2]. Number of patients with negative signs: ", patient_list.count(2))
            # print("2. Number of patient negative samples list for training", self._type_list.count(8)+self._type_list.count(9)+self._type_list.count(10)+self._type_list.count(11))
            # print(" 2-1. wimg-wtxt-pn: ", self._type_list.count(8))
            # print(" 2-2. wimg-w/otxt-pn: ", self._type_list.count(9))
            # print(" 2-3. w/oimg-wtxt-pn: ", self._type_list.count(10))
            # print(" 2-4. w/oimg-w/otxt-pn: ", self._type_list.count(11))
            print("[3]. Number of non-patients: ", patient_list.count(0))
            print("3. Number of non-patient negative samples for training", self._type_list.count(4)+self._type_list.count(5)+self._type_list.count(6)+self._type_list.count(7))
            print(" 3-1. wimg-wtxt-nn: ", self._type_list.count(4))
            print(" 3-2. wimg-w/otxt-nn: ", self._type_list.count(5))
            print(" 3-3. w/oimg-wtxt-nn: ", self._type_list.count(6))
            print(" 3-4. w/oimg-w/otxt-nn: ", self._type_list.count(7))
            # print("[1]. Number of patients: ", patient_list.count(1))
            # print("1. Number of patient positive samples for training", self._type_list.count(0)+self._type_list.count(3)+self._type_list.count(6)+self._type_list.count(9))
            # print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
            # print(" 1-2. wimg-w/otxt_pp: ", self._type_list.count(3))
            # print(" 1-3. w/oimg-wtxt_pp: ", self._type_list.count(6))
            # print(" 1-4. w/oimg-w/otxt_pp: ", self._type_list.count(9))
            # print("[2]. Number of patients with negative signs: ", patient_list.count(2))
            # print("2. Number of patient negative samples for training", self._type_list.count(1)+self._type_list.count(4)+self._type_list.count(7)+self._type_list.count(10))
            # print(" 2-1. wimg-wtxt-pn: ", self._type_list.count(1))
            # print(" 2-2. wimg-w/otxt-pn: ", self._type_list.count(4))
            # print(" 2-3. w/oimg-wtxt-pn: ", self._type_list.count(7))
            # print(" 2-4. w/oimg-w/otxt-pn: ", self._type_list.count(10))
            # print("[3]. Number of non-patients: ", patient_list.count(0))
            # print("3. Number of non-patient negative samples for training", self._type_list.count(2)+self._type_list.count(5)+self._type_list.count(8)+self._type_list.count(11))
            # print(" 3-1. wimg-wtxt-nn: ", self._type_list.count(2))
            # print(" 3-2. wimg-w/otxt-nn: ", self._type_list.count(5))
            # print(" 3-3. w/oimg-wtxt-nn: ", self._type_list.count(8))
            # print(" 3-4. w/oimg-w/otxt-nn: ", self._type_list.count(11))
        print("########## Detail Data Info ##########")
        print("Positive time-points: ", positive_tpoints)
        print("Negative time-points: ", negative_tpoints)
        
        self.feature_mins = self.train_min
        self.feature_maxs = self.train_max
        
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        pkl_path, possible_indices_keys, labels_by_dict, win_size, target = self._data_list[index]
        type_list = self._type_list[index]
        
        file_name = pkl_path.split("/")[-1]
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

        windowIndex = self.window_size - 1
        selectedKey = possible_indices_keys[0]
        dataSequence, maskSequence, deltaSequence, inputLength = sequenceGenerator(args, selectedKey, win_size, windowIndex, data_pkl)
        f_indices = False
        
        if args.carry_back:
            initStartIdx = data_pkl['initStartIdx']
            for idx, i in enumerate(initStartIdx):
                k = i - (selectedKey-win_size) -1
                
                if (i <= selectedKey) and (i > selectedKey-win_size):
                    dataSequence[:k, idx] = dataSequence[k, idx]
                    
        sample_len = dataSequence.shape[0]
        final_seqs = torch.zeros(3, self.window_size, self.vslt_len)
        final_seqs[0].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(dataSequence, args.vslt_mask, axis = 1)))
        final_seqs[1].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(maskSequence, args.vslt_mask, axis = 1)))
        final_seqs[2].narrow(0, 0, sample_len).copy_(torch.Tensor(np.delete(deltaSequence, args.vslt_mask, axis = 1)))

        if target == 0:
            multi_target = self.neg_multi_target
        else:
            firstOnset = ceil(labels_by_dict[selectedKey][0][0] // self.intv_len)
            multi_target = [0] * (firstOnset) + [1] * (12 - firstOnset)
        multi_target = torch.tensor(multi_target)

        missing = [False]   # Missing modality list: [vital/lab, img, txt]
        
        if (("img" in args.input_types and "img1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion and type_list in [0,2,3,5] and "img" in args.input_types)) and ('cxr_input' in data_pkl):
            cxr_li = [i for i in data_pkl['cxr_input'] if i[0] <= selectedKey] 
            if not cxr_li and ('test-full' in args.modality_inclusion): 
                print("collate cxr error")
                exit(1)
            elif not cxr_li and ('test-missing' in args.modality_inclusion): 
                img = torch.zeros(self.image_size).unsqueeze(0)
                missing.append(True)
            else:
                cxr_path = sorted(cxr_li)[-1][1]
                image = Image.open(self.image_data_path + cxr_path)
                image = F_t.equalize(image)
                img = self.transform(image)
                missing.append(False)
        else:
            img = torch.zeros(self.image_size).unsqueeze(0)
            missing.append(True)
        
        if (("txt" in args.input_types and "txt1" in args.fullmodal_definition and 'test-full' in args.modality_inclusion) or ('test-missing' in args.modality_inclusion and "txt" in args.input_types)) and ("txt1" in file_name):
            tokens = self.txtDict[(int(data_pkl['pat_id']), int(data_pkl['chid']))]
            if len(tokens) == 0:
                tokens = torch.zeros(self.token_max_length)
                textLength = 0
                missing.append(True)
            else:
                textLength = len(tokens)
                # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
                # EX) 2 {Sentence Tokens} {1 Padding} 3
                # Add Beginnning of Sentence Token
                tokens.insert(0, 2)
                tokens = torch.Tensor(clinical_note_transform(tokens))
                tokens[tokens==1] = 0
                missing.append(False)
        else:    
            tokens = torch.zeros(self.token_max_length)
            textLength = 0
            missing.append(True)
        missing = torch.Tensor(missing)
        return final_seqs, static_inputs, multi_target, inputLength, img, tokens, textLength, missing, f_indices

class SelfSupervisedLearning_dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Training Dataset for Self-Supervised Learning...")
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        lengths = []
        
        self.feature_mins = None
        self.feature_maxs = None
        self.transform = None
        self.txtDict = None
        self.vslt_len = len(args.vitalsign_labtest)
        self.intv_len = int(args.prediction_range // 12)
        self.window_size = args.window_size
        self.image_size = [args.image_size, args.image_size]
        self.input_types = args.input_types
        self.image_data_path = args.image_data_path
        self.token_max_length = args.bert_token_max_length
        self.txtDict = txtDictLoad("train")
        n_modality = len(args.input_types.split("_")) 
        
        # real-time x-ray image transform function
        if ("img" in self.input_types or 'train-missing' in args.modality_inclusion):
            self.transform = xray_image_transform_train()
                
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                
            # Exclude image based on the aspect ratio configuration
            if "cxr_input" in data_info:
                new_cxr_inputs = [cxr for cxr in data_info["cxr_input"] if float(cxr[1].split("_")[-1].split(".")[0]) >= args.ar_lowerbound and float(cxr[1].split("_")[-1].split(".")[0]) <= args.ar_upperbound]
                if len(new_cxr_inputs) > 0:
                    data_info["cxr_input"] = new_cxr_inputs
                else:
                    del data_info['cxr_input']
                    file_name = file_name.replace("_img1", "_img0")
                    
            # if full-modal setting, exclude missing modal file and if missing-modal setting, include all data
            if 'train-full' in args.modality_inclusion:
                if args.fullmodal_definition not in file_name:
                    continue
                if ("cxr_input" not in data_info and "img1" in args.fullmodal_definition):
                    continue
                if "txt1" in args.fullmodal_definition:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        continue
            else: # missing modality
                if "txt1" in file_name:
                    if (len(self.txtDict[(int(data_info['pat_id']), int(data_info['chid']))]) == 0): 
                        file_name = file_name.replace("_txt1_", "_txt0_")
                if ("cxr_input" not in data_info and "img1" in file_name):
                    file_name = file_name.replace("_img1", "_img0")
                
            # exclude if sequence length is less than the minimum reuiqred time-series length
            sequenceLength = data_info['data'].shape[0]
            if sequenceLength < (args.min_inputlen + args.prediction_range):
                continue            
            
            if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                continue
                
            possible_indices = list([i for i in range(args.min_inputlen-1, sequenceLength - args.prediction_range)])  # feasible data length
            
            if (("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion) or ('train-missing' in args.modality_inclusion)) and ('cxr_input' in data_info):
                earliest_img_time = min([j[0] for j in data_info['cxr_input']])
                possible_indices= list([i for i in possible_indices if earliest_img_time<=i])    
            
            if ("img1" in args.fullmodal_definition and 'train-full' in args.modality_inclusion):
                if not possible_indices: 
                    continue
            
            possibleWinSizes = data_info['possibleWinSizes']
            possible_indices = list([list(filter(lambda x: x in possibleWinSizes, key_list)) for key_list in possible_indices])
            
            if isListEmpty(possible_indices):
                continue
            
            self._data_list.append([pkl_path, possible_indices, possibleWinSizes])
            
            # if n_modality == 1:
            #     self._type_list.append(0)
            # elif n_modality == 2:
            #     sd
            # elif n_modality == 3:
            #     sdf
            # else:
            #     exit(1)
                            
            ######################################################            
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)

        self.feature_means = list(data_info['mean'])
        print("No Dataset Error: ", len(self._type_list) == len(self._data_list))
        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
        if ('train-full' in args.modality_inclusion):
            print("Number of patient positive samples list for training: {}".format(str(patient_list.count(1))))
        else: # missing modality
            print("[1]. Number of patients: ", patient_list.count(1))
            print("1. Number of patient positive samples lists for training", self._type_list.count(0)+self._type_list.count(1)+self._type_list.count(2)+self._type_list.count(3))
            print(" 1-1. wimg-wtxt_pp: ", self._type_list.count(0))
        
        self.feature_mins = self.train_min
        self.feature_maxs = self.train_max
        
        print("Dataset Prepared...\n")
    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input

#############################################################################################################################
################################################### Below Not Ready #############################################################
#############################################################################################################################

class Multitask_Training_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for Multitask_Training_Dataset Prediction...")
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []

        self._type_list = []
        possible_indices = {}

        lengths = []
        outbreak_types = ['death_time', 'vasso_time', 'cpr_time', 'intubation_time']

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal" or args.modality_inclusion == "trian-full_test-missing":
                if "img1" not in file_name:
                    continue
            """
            else:
                if "txt" in args.input_types:
                    if "txt1_" not in file_name:
                        continue
                elif "img" in args.input_types:
                    if "_img1" not in file_name:
                        continue
            """
                    
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                    continue
        
            if "img" in args.input_types and 'multi_task' in args.predict_type:
                # load cxr_img inputs
                cxr_list = data_info['cxr_input']   # list of (cxr_time, cxr_path)
                firstShot_time = cxr_list[0][0]
            
            #if "img" in args.input_types and 'cxr_input' not in data_info:#####
            #    continue
        
            ##### get possibles indices and max lengths here #####
            # If there are no positive cases in indices below args.min_inputlen, we change the case to negative.
            # If outbreak_time is beyond the prediction range of the given data or happened too early, change to 0 target
            target_list = []
            possible_indices_keys_all = [] 
            possible_indices_dict_all = []
            for outbreak_type in outbreak_types:
                outbreak_times = data_info[outbreak_type]
                if outbreak_times is not None and outbreak_type != "death_time"  and len(outbreak_times) != 0:
                    outbreak_times.sort()
                    if isinstance(outbreak_times[0], tuple):
                        outbreak_times = list([i for i in outbreak_times if (i[0] >= args.min_inputlen) and (i[0] <= sequenceLength + args.prediction_range - 1)])
                    else:
                        outbreak_times = list([i for i in outbreak_times if (i >= args.min_inputlen) and (i <= sequenceLength + args.prediction_range - 1)])
                    
                    if len(outbreak_times) == 0:
                        target = 0
                    else:
                        target = 1
                        
                elif outbreak_type == "death_time":
                    if(data_info['death_yn'] == 0):
                        target = 0
                    else:
                        if (outbreak_times > sequenceLength + args.prediction_range - 1) or (outbreak_times < args.min_inputlen):
                            target = 0
                        else:
                            target = 1 
                else:
                    target = 0

                # If target variable negative, get random subsequence of length 3 ~ args.window_size
                possible_indices_dict = {}            
                possible_indices_keys = []
    
                if target == 0:
                    outbreak_times = None
                    possible_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength)]  # feasible data length
                else:
                    # For within-n-hours task, the data must be within prediction_range of outbreak times
                    # range 2~3 means 2 < x =< 3
                    
                    if "death_time" == outbreak_type:
                        outbreak_times = math.ceil(outbreak_times)
                        possible_indices = [(outbreak_times - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_times >= args.min_inputlen+i-1) and (outbreak_times - i < sequenceLength)] or None
                        # print("possible_indices: ", possible_indices)
                        if possible_indices is None:
                            print("SeqLength : " + str(sequenceLength))
                            print(data_info)
                            raise Exception('Classification Error')
                    else:
                        dup_counts = []
                        possible_indices_list = []
                        for idx_outbreak, outbreak_time in enumerate(outbreak_times):
                            if isinstance(outbreak_times[0], tuple):
                                outbreak_time = math.ceil(outbreak_time[0])
                            else:
                                outbreak_time = math.ceil(outbreak_time)
                            if outbreak_time in dup_counts:
                                continue
                            else:
                                dup_counts.append(outbreak_time)
                            possible_indices = [(outbreak_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_time >= args.min_inputlen+i-1) and (outbreak_time - i < sequenceLength)] or None
                            
                            if possible_indices is None:
                                print("SeqLength : " + str(sequenceLength))
                                print(data_info)
                                raise Exception('Classification Error')
                            possible_indices_list.append(possible_indices)
                            # print(f"{str(idx_outbreak)}: ", possible_indices)
                        
                        # p_indices: [(32, [0, 1]), (31, [1, 2]), (30, [2, 3]), (29, [3, 4]), (28, [4, 5]), (27, [5, 6]), (26, [6, 7]), (25, [7, 8]), (24, [8, 9]), (23, [9, 10]), (22, [10, 11]), (21, [11, 12])]
                        for p_indices in possible_indices_list: 
                            # p_index: (32, [0, 1])
                            for p_index in p_indices:
                                if p_index[0] not in possible_indices_dict:
                                    possible_indices_dict[p_index[0]] = []
                                if p_index[1] not in possible_indices_dict[p_index[0]]:
                                    possible_indices_dict[p_index[0]].append(p_index[1])
                                if p_index[0] not in possible_indices_keys:
                                    possible_indices_keys.append(p_index[0])
                        possible_indices_keys.sort()
                        # print("possible_indices_keys: ", possible_indices_keys)
                        # print("possible_indices_dict: ", possible_indices_dict)
            
                target_list.append(target)
                if "img" in args.input_types and 'multi_task' in args.predict_type and data_info["cxr_input"] is not None:
                    possible_indices_keys = [key for key in possible_indices_keys if key >= firstShot_time]
                possible_indices_keys_all.append(possible_indices_keys)
                possible_indices_dict_all.append(possible_indices_dict)
            #     print("outbreak_times: ", outbreak_times)
            #     print("dict: ", possible_indices_dict)
            #     print("keys: ", possible_indices_keys)
            # print(max(target_list))
            # print(" ")
            
            final_possible_indices_dict = {}
            if max(target_list) == 0:
                final_possible_indices_keys = possible_indices_keys_all[0]
            else:
                final_possible_indices_keys = sorted(list(set([x for idx, xs in enumerate(possible_indices_keys_all) if target_list[idx] == 1 for x in xs])))
                for idx, key in enumerate(final_possible_indices_keys):
                    for t in range(len(outbreak_types)):
                        if target_list[t] == 0:
                            final_possible_indices_dict[key] = final_possible_indices_dict.get(key, list())
                            final_possible_indices_dict[key].append(None)
                        else:
                            if key in possible_indices_dict_all[t]:
                                final_possible_indices_dict[key] = final_possible_indices_dict.get(key, list())
                                final_possible_indices_dict[key].append(possible_indices_dict_all[t][key])
                            else:
                                final_possible_indices_dict[key] = final_possible_indices_dict.get(key, list())
                                final_possible_indices_dict[key].append(None)
                                
                all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength)]
                pat_neg_indices_keys = [item for item in all_indices_keys if item not in possible_indices_keys]
                if "img" in args.input_types and 'multi_task' in args.predict_type:
                    pat_neg_indices_keys = [key for key in pat_neg_indices_keys if key >= firstShot_time]

            # print("final_possible_indices_dict: ", final_possible_indices_dict)
            # print("final_possible_indices_keys: ", final_possible_indices_keys)
            # print(" ")
            # print("###########")
            # print(" ")
            if max(target_list) == 0 and (data_info['death_yn'] == 1 or data_info['vasso_yn'] == 1 or data_info['intubation_yn'] == 1 or data_info['cpr_yn'] == 1):            
                self._type_list.append(2)   
            else:
                self._type_list.append(max(target_list))   
            ######################################################
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)
                
            self._data_list.append([pkl_path, final_possible_indices_keys, final_possible_indices_dict, max(target_list)])
            
            if max(target_list) == 1 and len(pat_neg_indices_keys) > 0:
                self._data_list.append([pkl_path, pat_neg_indices_keys, {}, 0])
                self._type_list.append(2)
           
        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        print("Number of patient positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of patient negative samples for training: {}".format(str(self._type_list.count(2))))
        print("Number of non-patient negative samples for training: {}".format(str(self._type_list.count(0))))        
        print("Number of total negative samples for training: {}".format(str(self._type_list.count(0) + self._type_list.count(2))))    

        args.feature_mins = self.train_min
        args.feature_maxs = self.train_max

        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]
        
        return _input

class Multitask_Test_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for Multi-Task Prediction...")
        self._data_list = []
        _data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []

        self._type_list = []
        _type_list = []
        possible_indices = {}
        patDict = {}
        winDict = {}
        load_flag = False

        lengths = []
        outbreak_types = ['death_time', 'vasso_time', 'cpr_time', 'intubation_time']
        
        test_index_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__testset" + args.predict_type + "__" + args.modality_inclusion + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + ".pkl"
        test_winsize_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__testset" + args.predict_type + "__" + args.modality_inclusion + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + "_winsize.pkl"
        if not os.path.exists("./data/testIndexes"):
            os.makedirs('./data/testIndexes', exist_ok=True)
        if  os.path.exists(test_index_file) and os.path.exists(test_winsize_file) and data_type == "test dataset":
            # Open the file and add existing entries to dictionary
            print("Index file exists... Loading...")
            load_flag = True
            with open(test_index_file, 'rb') as f:
                patDict = pickle.load(f)
            with open(test_winsize_file, 'rb') as f:
                winDict = pickle.load(f)
        else:
            if os.path.exists(test_index_file):
                # shutil.rmtree(test_index_file)
                os.remove(test_index_file)
            if os.path.exists(test_winsize_file):
                # shutil.rmtree(test_winsize_file)
                os.remove(test_winsize_file)

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if "txt" in args.input_types:
                    if "txt1_" not in file_name:
                        continue
                elif "img" in args.input_types:
                    if "_img1" not in file_name:
                        continue
            """
                    
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            pat_id = int(data_info['pat_id'])
            chid = int(data_info['chid'])   
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                    continue
                
            if "img" in args.input_types and 'multi_task' in args.predict_type:
                # load cxr_img inputs
                cxr_list = data_info['cxr_input']   # list of (cxr_time, cxr_path)
                firstShot_time = cxr_list[0][0]
            
            #if "img" in args.input_types and 'cxr_input' not in data_info:#####
            #    continue
            #### Check!!!!####   
            # Check if the possible_indices_dict for the given patient has already been initialized
            if (pat_id, chid, 0) in patDict or (pat_id, chid, 1) in patDict:
                if (pat_id, chid, 0) in patDict:
                    possible_indices_keys, possible_indices_dict, target, target_type = patDict[(pat_id, chid, 0)]
                    if "img" in args.input_types and 'multi_task' in args.predict_type and data_info["cxr_input"] is not None:
                        possible_indices_keys = [key for key in possible_indices_keys if key >= firstShot_time]
                    _data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, target])
                    _type_list.append(target_type)
                    
                if (pat_id, chid, 1) in patDict:
                    possible_indices_keys, possible_indices_dict, target, target_type = patDict[(pat_id, chid, 1)]
                    if "img" in args.input_types and 'multi_task' in args.predict_type and data_info["cxr_input"] is not None:
                        possible_indices_keys = [key for key in possible_indices_keys if key >= firstShot_time]
                    _data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, target])
                    _type_list.append(target_type)
            else:
                ##### get possibles indices and max lengths here #####
                # If there are no positive cases in indices below args.min_inputlen, we change the case to negative.
                # If outbreak_time is beyond the prediction range of the given data or happened too early, change to 0 target
                target_list = []
                possible_indices_keys_all = [] 
                possible_indices_dict_all = []
                for outbreak_type in outbreak_types:
                    outbreak_times = data_info[outbreak_type]
                    if outbreak_times is not None and outbreak_type != "death_time" and len(outbreak_times) != 0:
                        outbreak_times.sort()
                        if isinstance(outbreak_times[0], tuple):                         
                            outbreak_times = list([i for i in outbreak_times if (i[0] >= args.min_inputlen) and (i[0] <= sequenceLength + args.prediction_range - 1)])
                        else:
                            outbreak_times = list([i for i in outbreak_times if (i >= args.min_inputlen) and (i <= sequenceLength + args.prediction_range - 1)])
                        
                        if len(outbreak_times) == 0:
                            target = 0
                        else:
                            target = 1
                            
                    elif outbreak_type == "death_time":
                        if(data_info['death_yn'] == 0):
                            target = 0
                        else:
                            if (outbreak_times > sequenceLength + args.prediction_range - 1) or (outbreak_times < args.min_inputlen):
                                target = 0
                            else:
                                target = 1 
                    else:
                        target = 0

                    # If target variable negative, get random subsequence of length 3 ~ args.window_size
                    possible_indices_dict = {}            
                    possible_indices_keys = []
        
                    if target == 0:
                        outbreak_times = None
                        possible_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength)]  # feasible data length
                    else:
                        # For within-n-hours task, the data must be within prediction_range of outbreak times
                        # range 2~3 means 2 < x =< 3
                        
                        if "death_time" == outbreak_type:
                            outbreak_times = math.ceil(outbreak_times)
                            possible_indices = [(outbreak_times - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_times >= args.min_inputlen+i-1) and (outbreak_times - i < sequenceLength)] or None
                            # print("possible_indices: ", possible_indices)
                            if possible_indices is None:
                                print("SeqLength : " + str(sequenceLength))
                                print(data_info)
                                raise Exception('Classification Error')
                        else:
                            dup_counts = []
                            possible_indices_list = []
                            for idx_outbreak, outbreak_time in enumerate(outbreak_times):
                                if isinstance(outbreak_times[0], tuple):
                                    outbreak_time = math.ceil(outbreak_time[0])
                                else:
                                    outbreak_time = math.ceil(outbreak_time)
                                if outbreak_time in dup_counts:
                                    continue
                                else:
                                    dup_counts.append(outbreak_time)
                                possible_indices = [(outbreak_time - i, [i-1, i]) for i in range(1, args.prediction_range + 1) if (outbreak_time >= args.min_inputlen+i-1) and (outbreak_time - i < sequenceLength)] or None
                                
                                if possible_indices is None:
                                    print("SeqLength : " + str(sequenceLength))
                                    print(data_info)
                                    raise Exception('Classification Error')
                                possible_indices_list.append(possible_indices)
                                # print(f"{str(idx_outbreak)}: ", possible_indices)
                            
                            # p_indices: [(32, [0, 1]), (31, [1, 2]), (30, [2, 3]), (29, [3, 4]), (28, [4, 5]), (27, [5, 6]), (26, [6, 7]), (25, [7, 8]), (24, [8, 9]), (23, [9, 10]), (22, [10, 11]), (21, [11, 12])]
                            for p_indices in possible_indices_list: 
                                # p_index: (32, [0, 1])
                                for p_index in p_indices:
                                    if p_index[0] not in possible_indices_dict:
                                        possible_indices_dict[p_index[0]] = []
                                    if p_index[1] not in possible_indices_dict[p_index[0]]:
                                        possible_indices_dict[p_index[0]].append(p_index[1])
                                    if p_index[0] not in possible_indices_keys:
                                        possible_indices_keys.append(p_index[0])
                            possible_indices_keys.sort()
                            # print("possible_indices_keys: ", possible_indices_keys)
                            # print("possible_indices_dict: ", possible_indices_dict)
                
                    target_list.append(target)
                    if "img" in args.input_types and 'multi_task' in args.predict_type and data_info["cxr_input"] is not None:
                        possible_indices_keys = [key for key in possible_indices_keys if key >= firstShot_time]
                    possible_indices_keys_all.append(possible_indices_keys)
                    possible_indices_dict_all.append(possible_indices_dict)
                #     print("outbreak_times: ", outbreak_times)
                #     print("dict: ", possible_indices_dict)
                #     print("keys: ", possible_indices_keys)
                # print(max(target_list))
                # print(" ")
                
                final_possible_indices_dict = {}
                if max(target_list) == 0:
                    final_possible_indices_keys = possible_indices_keys_all[0]
                    
                    if (data_info['death_yn'] == 1 or data_info['vasso_yn'] == 1 or data_info['intubation_yn'] == 1 or data_info['cpr_yn'] == 1):
                        target_type = 2
                        if len(final_possible_indices_keys) >= args.PatNegSampleN:
                            final_possible_indices_keys = random.sample(final_possible_indices_keys, args.PatNegSampleN)
                        _data_list.append([pkl_path, final_possible_indices_keys, final_possible_indices_dict, max(target_list)])
                        _type_list.append(target_type)
                        if data_type == "test dataset":
                            patDict[(pat_id, chid, 0)] = final_possible_indices_keys, final_possible_indices_dict, target, target_type
                    else:
                        target_type = 0
                        if len(final_possible_indices_keys) >= args.nonPatNegSampleN:
                            final_possible_indices_keys = random.sample(final_possible_indices_keys, args.nonPatNegSampleN)
                        _data_list.append([pkl_path, final_possible_indices_keys, final_possible_indices_dict, max(target_list)])
                        _type_list.append(target_type)
                        if data_type == "test dataset":
                            patDict[(pat_id, chid, 0)] = final_possible_indices_keys, final_possible_indices_dict, target, target_type
                else:
                    final_possible_indices_keys = sorted(list(set([x for idx, xs in enumerate(possible_indices_keys_all) if target_list[idx] == 1 for x in xs])))
                    for idx, key in enumerate(final_possible_indices_keys):
                        for t in range(len(outbreak_types)):
                            if target_list[t] == 0:
                                final_possible_indices_dict[key] = final_possible_indices_dict.get(key, list())
                                final_possible_indices_dict[key].append(None)
                            else:
                                if key in possible_indices_dict_all[t]:
                                    final_possible_indices_dict[key] = final_possible_indices_dict.get(key, list())
                                    final_possible_indices_dict[key].append(possible_indices_dict_all[t][key])
                                else:
                                    final_possible_indices_dict[key] = final_possible_indices_dict.get(key, list())
                                    final_possible_indices_dict[key].append(None)
                                    
                    all_indices_keys = [i for i in range(args.min_inputlen-1, sequenceLength)]
                    final_pat_neg_indices_keys = [item for item in all_indices_keys if item not in final_possible_indices_keys]
                    if "img" in args.input_types and 'multi_task' in args.predict_type and data_info["cxr_input"] is not None:
                        final_pat_neg_indices_keys = [key for key in final_pat_neg_indices_keys if key >= firstShot_time]
                        
                    if len(final_pat_neg_indices_keys) >= args.PatNegSampleN:
                        final_pat_neg_indices_keys = random.sample(final_pat_neg_indices_keys, args.PatNegSampleN)

                    if len(final_possible_indices_keys) >= args.PatPosSampleN:
                        final_possible_indices_keys = random.sample(final_possible_indices_keys, args.PatPosSampleN)
                    
                    if data_type == "test dataset":
                        patDict[(pat_id, chid, 1)] = [final_possible_indices_keys, final_possible_indices_dict, 1, 1] 
                        patDict[(pat_id, chid, 0)] = [final_pat_neg_indices_keys, {}, 0, 2] 
                        
                    _data_list.append([pkl_path, final_possible_indices_keys, final_possible_indices_dict, 1])
                    _type_list.append(1)
                    
                    if len(final_pat_neg_indices_keys) > 0:
                        _data_list.append([pkl_path, final_pat_neg_indices_keys, {}, 0])
                        _type_list.append(2)

                # print("final_possible_indices_dict: ", final_possible_indices_dict)
                # print("final_possible_indices_keys: ", final_possible_indices_keys)
                # print(" ")
                # print("###########")
                # print(" ")
            ######################################################
            
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)
        
        if data_type == "test dataset" and load_flag == False:
            with open(test_index_file, 'wb') as f:
                pickle.dump(patDict, f, pickle.HIGHEST_PROTOCOL)
            
        self.feature_means = list(data_info['mean'])
        
        for idx, sample in enumerate(_data_list):
            pkl_pth, p_keys, p_dict, t = sample
            t_type = _type_list[idx]
            for key in p_keys:
                win_key_name = "_".join(pkl_pth.split("/")[-1].split("_")[:2]) + f"_{str(key)}"
                if win_key_name in winDict:     
                    win_size = winDict[win_key_name]
                else:
                    win_max = args.window_size if key+2 >= args.window_size else key+2   
                    win_size = random.randrange(args.min_inputlen, win_max)
                    winDict[win_key_name] = win_size
                self._data_list.append([pkl_pth, [key], p_dict, t, win_size])
                self._type_list.append(t_type)

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        print("Number of patient positive samples for training: {}".format(str(_type_list.count(1))))
        print("Number of patient negative samples for training: {}".format(str(_type_list.count(2))))
        print("Number of non-patient negative samples for training: {}".format(str(_type_list.count(0))))        
        print("Number of total negative samples for training: {}".format(str(_type_list.count(0) + _type_list.count(2))))    

        args.feature_mins = self.train_min
        args.feature_maxs = self.train_max

        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]
        
        return _input

#############################################################################################################################
################################################### Old Dataset #############################################################
#############################################################################################################################

class Binary_uni_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for Binary_uni_Dataset Prediction...")
        print(f'Dataset for Task: {args.input_types}')
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        lengths = []
        tmpTasks = ['mortality', 'vasso', 'intubation', 'cpr']
        tmpYN = ['death_yn', 'vasso_yn', 'intubation_yn', 'cpr_yn']
        tmpInputs = ['death_time', 'vasso_time', 'intubation_time', 'cpr_time']
        taskIndex = tmpTasks.index(args.output_type) 
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal" or args.modality_inclusion == "trian-full_test-missing":
                if "img1" not in file_name:
                    continue
            """    
            else:
                if "txt" in args.input_types:
                    if "txt1_" not in file_name:
                        continue
                elif "img" in args.input_types:
                    if "_img1" not in file_name:
                        continue
            """

            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            # Q: all vitalsign_labtest features are required?
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                    continue
                
            # TimeBetweenAdmVital: vital sign not recorded time between admission time and vitial sign measuring starting time
            TimeBetweenAdmVital =  round(data_info['window_first_idx_hr'] - DatetimeToHours(data_info['admission_time']))
            if TimeBetweenAdmVital > (args.window_size - args.min_inputlen) or TimeBetweenAdmVital < 0:
                # if TimeBetweenAdmVital < 0:
                continue
            
            ##### get possibles indices and max lengths here #####
            # target_Aftertype is one of:
            # target = 0: non patient
            # target = 1: patient with outbreak soon
            # target = 2: patient with outbreak on far future or far past but currently in normal group
            # targetAfterRange is: pos/neg lable of current sample
            
            # Q: What are these? (possible_indices)
            possible_indices_dict = {}            
            possible_indices_keys = []
            if args.output_type == "mortality":
                # negative patient: no death event                  
                if(data_info['death_yn'] == 0):
                    target_Aftertype = 0
                    targetAfterRange = 0

                # positive patient: death event occured               
                else:
                    death_time = data_info['death_time']
                    # exclude: invalid death time (admission - death - first V/S data point)
                    if death_time <= (args.window_size - TimeBetweenAdmVital - 1):
                        continue
                    # If death time is beyond the prediction range of the given data, 
                    # change target to 0 (target_Aftertype: 2)
                    if (death_time > (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1)):
                        target_Aftertype = 2
                        targetAfterRange = 0
                        # here we get window end-index of max length within 24 hours
                        possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]
                    # otherwise, the sample remains to target 1
                    else:                    
                        target_Aftertype = 1
                        targetAfterRange = 1
                        
            else: # output: vasso, intubation, cpr
                if(data_info[tmpYN[taskIndex]] == 0):
                    target_Aftertype = 0
                    targetAfterRange = 0
                else:
                    outbreak_times = data_info[tmpInputs[taskIndex]]
                    # TO-DO: check this on preprocess stage - it should have be negative patient already
                    if len(outbreak_times) == 0 or outbreak_times == None:
                        target_Aftertype = 0
                        targetAfterRange = 0
                    else:
                        if isinstance(outbreak_times[0], tuple): # event_time: tuple list (start, end) 
                            outbreak_bool = any([True if (i[0] <= (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1) and\
                                                 i[0] > (args.window_size - TimeBetweenAdmVital - 1)) else False for i in outbreak_times])
                        else:   # event_time: list
                            outbreak_bool = any([True if (i <= (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1) and\
                                                 i > (args.window_size - TimeBetweenAdmVital - 1)) else False for i in outbreak_times])
                            
                        # If outbreak time is beyond the prediction range of the given data or happened too early, change to 0 target
                        if outbreak_bool:
                            target_Aftertype = 1
                            targetAfterRange = 1
                        else:                    
                            target_Aftertype = 2
                            targetAfterRange = 0
                ######################################################
            
            if sum(data_info['vsMissSteps'][:possible_indices_keys[0]+1]) >= (possible_indices_keys[0]+1)//3:
                self._data_list.append([pkl_path, possible_indices_keys, possible_indices_dict, targetAfterRange])
                self._type_list.append(target_Aftertype)
            else:
                # print(data_info['mask'][:possible_indices_keys[0]+1,:])
                continue
            
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)
            
                
        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        # else:
        #     self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]
        
        print("Number of patient positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of patient negative samples for training: {}".format(str(self._type_list.count(2))))
        print("Number of non-patient negative samples for training: {}".format(str(self._type_list.count(0))))        
        print("Number of total negative samples for training: {}".format(str(self._type_list.count(0) + self._type_list.count(2))))        
        
        args.feature_mins = self.train_min
        args.feature_maxs = self.train_max
        
        print("Dataset Prepared...\n")


    def __repr__(self):
        return (f"Data path: {self._data_pkl}")


    def __len__(self):
        return len(self._data_list)


    def __getitem__(self, index):
        return self._data_list[index]

class Binary_uniTest_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Training Dataset for Binary_uniTraining_Dataset Prediction...")
        print(f'Dataset for Task: {args.input_types}')
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []
        self._type_list = []
        lengths = []
        tmpTasks = ['mortality', 'vasso', 'intubation', 'cpr']
        tmpYN = ['death_yn', 'vasso_yn', 'intubation_yn', 'cpr_yn']
        tmpInputs = ['death_time', 'vasso_time', 'intubation_time', 'cpr_time']
        taskIndex = tmpTasks.index(args.output_type) 
        patDict = {}
        
        load_flag = False
        
        test_index_file = "./data/testIndexes/testIndexes__" + args.test_data_path.split("/")[-2] + "__" + args.modality_inclusion + "__winsize" + str(args.window_size) + "__minlen" + str(args.min_inputlen) + "__" + args.output_type + "__PW" + str(args.prediction_range) + ".txt"
        if not os.path.exists("./data/testIndexes"):
            os.makedirs('./data/testIndexes', exist_ok=True)
        if  os.path.exists(test_index_file):
            print("Index file exists... Loading...")
            load_flag = True
            # File Format : {pat_id} {chid} {randIndex}
            # Open the file and add existing entries to dictionary
            if exists(test_index_file):
                indexFile = open(test_index_file, "r")
                while True:
                    line = indexFile.readline()
                    if not line:
                        break
                    line = line.strip().split()
                    pat_id = int(line[0])
                    chid = int(line[1])
                    windowEndIndex = int(line[2])
                    predWithin = int(line[3])
                    target_Aftertype = int(line[4])
                    targetAfterRange = int(line[5])
                    target_AllType = int(line[6])
                    targetAllRange = int(line[7])
                    patDict[(pat_id, chid)] = [windowEndIndex, predWithin, target_Aftertype, targetAfterRange, target_AllType, targetAllRange] # targetAfterRange for uni-img and uni-vs; targetAllRange for uni-txt

                indexFile.close()
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if "txt" in args.input_types:
                    if "txt1_" not in file_name:
                        continue
                elif "img" in args.input_types:
                    if "_img1" not in file_name:
                        continue
            """

            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            sequenceLength = data_info['data'].shape[0]
            # We need args.min_inputlen data points (hours) in the features,
            if sequenceLength < args.min_inputlen:
                continue
            
            # if features from args.vitalsign_labtest are not prepared, exclude
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                    continue
                
            # TimeBetweenAdmVital: vital sign not recorded time between admission time and vitial sign measuring starting time
            TimeBetweenAdmVital =  round(data_info['window_first_idx_hr'] - DatetimeToHours(data_info['admission_time']))
            if TimeBetweenAdmVital > (args.window_size - args.min_inputlen) or TimeBetweenAdmVital < 0:
                continue
                
            ##### get possibles indices and max lengths here #####
            # target = 0: non patient
            # target = 1: patient with outbreak soon
            # target = 2: patient with outbreak on far future or far past but currently in normal group
            possible_indices_dict = {}            
            possible_indices_keys = []
            pat_id = int(data_info['pat_id'])
            chid = int(data_info['chid'])
            
            if (pat_id, chid) in patDict:
                windowEndIndex, predWithin, target_Aftertype, targetAfterRange, target_AllType, targetAllRange = patDict[(pat_id, chid)]

            else:
                outputFile = open(test_index_file, 'a')
                if args.output_type == "mortality":                    
                    if(data_info['death_yn'] == 0):
                        target_Aftertype = 0
                        targetAfterRange = 0
                        target_AllType = 0
                        targetAllRange = 0
                        possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours

                    else:
                        death_time = data_info['death_time']
                        if death_time <= (args.window_size - TimeBetweenAdmVital - 1):
                            continue
                        # If death time is beyond the prediction range of the given data or happened too early, change to 0 target
                        if death_time > (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1):
                            target_Aftertype = 2
                            targetAfterRange = 0
                            possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours

                        else:                    
                            target_Aftertype = 1
                            targetAfterRange = 1
                            possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours
                            
                        if (death_time + TimeBetweenAdmVital) <= 0:
                            continue
                        if (death_time > (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1)):
                            target_AllType = 2
                            targetAllRange = 0
                        else:
                            target_AllType = 1
                            targetAllRange = 1
                            
                else:
                    if(data_info[tmpYN[taskIndex]] == 0):
                        target_Aftertype = 0
                        targetAfterRange = 0
                        target_AllType = 0
                        targetAllRange = 0
                        possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours

                    else:
                        outbreak_times = data_info[tmpInputs[taskIndex]]
                        if len(outbreak_times) == 0 or outbreak_times == None:
                            target_Aftertype = 0
                            targetAfterRange = 0
                            target_AllType = 0
                            targetAllRange = 0
                            possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours
                        else:
                    
                            if isinstance(outbreak_times[0], tuple):
                                outbreak_bool = any([True if (i[0] <= (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1) and i[0] > (args.window_size - TimeBetweenAdmVital - 1)) else False for i in outbreak_times ])
                                outbreak_bool_allRange = any([True if (i[0] <= (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1) and i[0]> 0 - TimeBetweenAdmVital) else False for i in outbreak_times ])

                            else:
                                outbreak_bool = any([True if (i <= (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1) and i > (args.window_size - TimeBetweenAdmVital - 1)) else False for i in outbreak_times ])
                                outbreak_bool_allRange = any([True if (i <= (args.window_size + args.prediction_range - TimeBetweenAdmVital - 1) and i > 0 - TimeBetweenAdmVital) else False for i in outbreak_times ])
                                
                            # If outbreak time is beyond the prediction range of the given data or happened too early, change to 0 target
                            if outbreak_bool:
                                target_Aftertype = 1
                                targetAfterRange = 1
                                possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours

                            else:                    
                                target_Aftertype = 2
                                targetAfterRange = 0
                                possible_indices_keys = [args.window_size - TimeBetweenAdmVital - 1]  # here we get window end-index of max length within 24 hours
                            
                            if outbreak_bool_allRange:
                                target_AllType = 1
                                targetAllRange = 1
                            else:
                                target_AllType = 2
                                targetAllRange = 0
                                
                windowEndIndex = possible_indices_keys[0]
                predWithin = args.prediction_range
                outputFile.write(f"{str(int(pat_id))} {str(int(chid))} {str(int(windowEndIndex))} {str(int(args.prediction_range))} {str(int(target_Aftertype))} {str(int(targetAfterRange))} {str(int(target_AllType))} {str(int(targetAllRange))}\n")
                outputFile.close()
                
            ######################################################
            
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(sequenceLength)
                
            if windowEndIndex+1 < 24:
                winStartIndex = 0
            else:
                winStartIndex = windowEndIndex - 23
            # if sum(data_info['vsMissSteps'][winStartIndex:windowEndIndex+1]) >= 3:
            if sum(data_info['vsMissSteps'][:windowEndIndex+1]) >= (windowEndIndex+1)//3:
                self._data_list.append([pkl_path, [windowEndIndex], possible_indices_dict, targetAfterRange])
                self._type_list.append(target_Aftertype)
            else:
                # print(data_info['mask'][:possible_indices_keys[0]+1,:])
                continue
            # self._data_list.append([pkl_path, [windowEndIndex], possible_indices_dict, targetAfterRange])            
            # self._type_list.append(target_Aftertype)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        # else:
        #     self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]
        
        print("Number of patient positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of patient negative samples for training: {}".format(str(self._type_list.count(2))))
        print("Number of non-patient negative samples for training: {}".format(str(self._type_list.count(0))))        
        print("Number of total negative samples for training: {}".format(str(self._type_list.count(0) + self._type_list.count(2))))        
        
        
        args.feature_mins = self.train_min
        args.feature_maxs = self.train_max
        
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input

### currently possible window sizes are not working for "seq2seq" and "mutlitask_training/test" datasets

class txt_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        print("Preparing Text Dataset...")
        self._data_list = []
        self._type_list = []

        self.train_min = []
        self.train_max = []
        self.feature_means = []

        for idx, txtInput in enumerate(tqdm(data, desc="Loading Text of {}...".format(data_type))):
            self._data_list.append(txtInput)

            positivities = txtInput.split("/")[-1].split()
            if args.output_type == "mortality":
                self._type_list.append(int(positivities[0]))
            elif args.output_type == "vasso":
                self._type_list.append(int(positivities[2]))
            elif args.output_type == "cpr":
                self._type_list.append(int(positivities[4]))
            elif args.output_type == "intubation":
                self._type_list.append(int(positivities[6]))
            elif args.output_type == "all":
                self._type_list.append(int(int(positivities[0]) or int(positivities[2]) or int(positivities[4]) or int(positivities[6])))
            else:
                raise NotImplementedError
    
    def __repr__(self):
        return (f"Data path: {self._data_pkl}")
    
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, index):
        _input = self._data_list[index]
        return _input

class img_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type='dataset'):
        print('Preparing Dataset for Image dataset...')
        self._data_list = []    # (X, y)
        self._type_list = []    # (target type: 0,1,2)
        
        # output list
        tmpTasks  = ['mortality', 'vasso', 'intubation', 'cpr']
        tmpYN     = ['death_yn', 'vasso_yn', 'intubation_yn', 'cpr_yn']
        tmpInputs = ['death_time', 'vasso_time', 'intubation_time', 'cpr_time']
        taskIndex = tmpTasks.index(args.output_type) 

        # load pkl data (list)
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if "txt" in args.input_types:
                    if "txt1_" not in file_name:
                        continue
                elif "img" in args.input_types:
                    if "_img1" not in file_name:
                        continue
            """

            # load current pkl data
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)

            # get ICU admission, discharge time
            admission_time, discharge_time = data_info['admission_time'], data_info['discharge_time']           # datetime
            admission_time, discharge_time = DatetimeToHours(admission_time), DatetimeToHours(discharge_time)   # hours (float)
            
            # load cxr_img inputs
            cxr_list = data_info['cxr_input']   # list of (cxr_time, cxr_path)
            # select the most recent cxr img in 24hrs from admission
            # we can select cxr img before admission, if there is no cxr img after admission
            cxr_list = [cxr_record for cxr_record in cxr_list if cxr_record[0] < (admission_time + 24 - 1)]
            cxr_list = sorted(cxr_list, key=lambda x: x[0]) # sort by cxr_time order
            
            # check: if there is valid cxr img
            if len(cxr_list) == 0:
                continue
            
            ###
            # sequenceLength = data_info['data'].shape[0]
            # if sequenceLength < args.min_inputlen:
            #     continue
            
            # # if features from args.vitalsign_labtest are not prepared, exclude
            # # Q: all vitalsign_labtest features are required?
            # if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
            #     continue                
            
            # TimeBetweenAdmVital =  round(data_info['window_first_idx_hr'] - DatetimeToHours(data_info['admission_time']))
            # if TimeBetweenAdmVital > (args.window_size - args.min_inputlen) or TimeBetweenAdmVital < 0:
            #     # if TimeBetweenAdmVital < 0:
            #     continue
            ###
            
            # check: if there is cxr img during ICU stay, then discard cxr imgs before ICU admission
            last_cxr_time, _ = cxr_list[-1] # get the latest cxr record
            if last_cxr_time >= admission_time:
                cxr_list = [cxr_record for cxr_record in cxr_list if cxr_record[0] >= admission_time]
                cxr_time, cxr_path  = cxr_list[0]  # select the oldest (close to admission time) item
            else:
                cxr_time, cxr_path  = cxr_list[-1]  # select the latest (close to admission time) item

            # load jpeg image, rescale (0~1), convert to PIL image 
            # IMG_DIR             = '/nfs/thena/MedicalAI/ImageData/public/MIMIC_CXR/data' 
            # img_path            = os.path.join(IMG_DIR, cxr_path)
            # image               = Image.open(img_path)
            # image               = np.array(image) / 255. # rescale 0~1
            # image               = Image.fromarray(image)

            # check targets - mortality
            if args.output_type == 'mortality':
                # negative patient: no death event
                if data_info['death_yn'] == 0:
                    target_Aftertype = 0
                    targetAfterRange = 0
                # positive patient: death event occured
                else:
                    death_time = data_info['death_time']
                    # exclude: invalid death time (death - cxr)
                    if death_time <= cxr_time:
                        continue
                    # If death time is beyond the prediction range of the given data,
                    # change target to 0 (target_Aftertype: 2)
                    if death_time > (cxr_time + args.prediction_range - 1):
                        target_Aftertype = 2
                        targetAfterRange = 0
                    else:
                        target_Aftertype = 1
                        targetAfterRange = 1
            # check targets - vasso, intubation, cpr
            else: 
                # negative sample: no event
                if data_info[tmpYN[taskIndex]] == 0:
                    target_Aftertype = 0
                    targetAfterRange = 0
                # positive sample: event occured
                else:
                    outbreak_times = data_info[tmpInputs[taskIndex]]
                    outbreak_times = [i for i in outbreak_times if i[0] >= 0]
                    # TO-DO: check this on preprocess stage - it should have be negative patient already
                    if len(outbreak_times) == 0 or outbreak_times == None:
                        target_Aftertype = 0
                        targetAfterRange = 0
                    else:
                        if isinstance(outbreak_times[0], tuple): # event_time: tuple list (start, end) 
                            outbreak_bool = any([True if (i[0] <= (cxr_time + args.prediction_range - 1) and\
                                                            i[0] > (cxr_time - 1)) else False for i in outbreak_times])
                        else:   # event_time: list
                            outbreak_bool = any([True if (i <= (cxr_time + args.prediction_range - 1) and\
                                                            i > (cxr_time - 1)) else False for i in outbreak_times])
                        # If outbreak time is beyond the prediction range of the given data,
                        # or happened too early, change to 0 target
                        if outbreak_bool:
                            target_Aftertype = 1
                            targetAfterRange = 1
                        else:                    
                            target_Aftertype = 2
                            targetAfterRange = 0
            
            # save input & output to list
            # self._data_list.append([image, targetAfterRange])
            self._data_list.append([cxr_path, targetAfterRange])
            self._type_list.append(target_Aftertype)

        print("Number of patient positive samples: {}".format(str(self._type_list.count(1))))
        print("Number of patient negative samples: {}".format(str(self._type_list.count(2))))
        print("Number of non-patient negative samples: {}".format(str(self._type_list.count(0))))        
        print("Number of total negative samples: {}".format(str(self._type_list.count(0) + self._type_list.count(2))))        
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")


    def __len__(self):
        return len(self._data_list)


    def __getitem__(self, index):
        return self._data_list[index]

class img_Dataset_r(torch.utils.data.Dataset):
    # random start
    def __init__(self, args, data, data_type='dataset'):
        print('Preparing Dataset for Image dataset...')
        print("random start point for unimodal image structure...")
        self._data_list = []    # (X, y)
        self._type_list = []    # (target type: 0,1,2)
        
        # output list
        tmpTasks  = ['mortality', 'vasso', 'intubation', 'cpr']
        tmpYN     = ['death_yn', 'vasso_yn', 'intubation_yn', 'cpr_yn']
        tmpInputs = ['death_time', 'vasso_time', 'intubation_time', 'cpr_time']
        taskIndex = tmpTasks.index(args.output_type) 

        # load pkl data (list)
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal" or data_type == 'test dataset':  # fullmodal data inclusion check (1)
                if "img1" not in file_name:
                    continue
            """
            else:
                if "txt" in args.input_types:
                    if "txt1_" not in file_name:
                        continue
                elif "img" in args.input_types:
                    if "_img1" not in file_name:
                        continue
            """

            # load current pkl data
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            if args.modality_inclusion == "fullmodal" or data_type == 'test dataset':  # fullmodal data inclusion check (2)
                sequenceLength = data_info['data'].shape[0]
                if sequenceLength < args.min_inputlen:
                    continue
                if not all([True if i in data_info['feature_order'] else False for i in args.mandatory_vitalsign_labtest]):
                    continue

            # get ICU admission, discharge time
            admission_time, discharge_time = data_info['admission_time'], data_info['discharge_time']           # datetime
            admission_time, discharge_time = DatetimeToHours(admission_time), DatetimeToHours(discharge_time)   # hours (float)
            
            # load cxr_img inputs
            cxr_list = data_info['cxr_input']   # list of (cxr_time, cxr_path)
            # get all cxr_inputs after first v/s data point 
            cxr_list = [cxr_record for cxr_record in cxr_list if cxr_record[0] >= 0]
            cxr_list = sorted(cxr_list, key=lambda x: x[0]) # sort by cxr_time order
            
            # check: if there is valid cxr img
            if len(cxr_list) == 0:
                continue
            
            for cxr in cxr_list:
                cxr_time, cxr_path = cxr
                # check targets - mortality
                if args.output_type == 'mortality':
                    # negative patient: no death event
                    if data_info['death_yn'] == 0:
                        target_Aftertype = 0
                        targetAfterRange = 0
                    # positive patient: death event occured
                    else:
                        death_time = data_info['death_time']
                        # exclude: invalid death time 
                        if death_time <= 0:
                            continue
                        # If death time is beyond the prediction range of the given data,
                        # change target to 0 (target_Aftertype: 2)
                        if death_time > (cxr_time + args.prediction_range - 1):
                            target_Aftertype = 2
                            targetAfterRange = 0
                        else:
                            target_Aftertype = 1
                            targetAfterRange = 1
                        # if death_time <= (cxr_time + args.prediction_range - 1) and death_time > cxr_time - 1:
                        #     target_Aftertype = 1
                        #     targetAfterRange = 1
                        # else:
                        #     target_Aftertype = 2
                        #     targetAfterRange = 0
                # check targets - vasso, intubation, cpr
                else: 
                    # negative sample: no event
                    if data_info[tmpYN[taskIndex]] == 0:
                        target_Aftertype = 0
                        targetAfterRange = 0
                    # positive sample: event occured
                    else:
                        outbreak_times = data_info[tmpInputs[taskIndex]]
                        # TO-DO: check this on preprocess stage - it should have be negative patient already
                        if len(outbreak_times) == 0 or outbreak_times == None:
                            target_Aftertype = 0
                            targetAfterRange = 0
                        else:
                            if isinstance(outbreak_times[0], tuple): # event_time: tuple list (start, end) 
                                outbreak_bool = any([True if (i[0] <= (cxr_time + args.prediction_range - 1) and\
                                                                i[0] > (cxr_time - 1)) else False for i in outbreak_times])
                            else:   # event_time: list
                                outbreak_bool = any([True if (i <= (cxr_time + args.prediction_range - 1) and\
                                                                i > (cxr_time - 1)) else False for i in outbreak_times])
                            # If outbreak time is beyond the prediction range of the given data,
                            # or happened too early, change to 0 target
                            if outbreak_bool:
                                target_Aftertype = 1
                                targetAfterRange = 1
                            else:                    
                                target_Aftertype = 2
                                targetAfterRange = 0
                
                # save input & output to list
                # self._data_list.append([image, targetAfterRange])
                self._data_list.append([cxr_path, targetAfterRange])
                self._type_list.append(target_Aftertype)

        print("Number of patient positive samples: {}".format(str(self._type_list.count(1))))
        print("Number of patient negative samples: {}".format(str(self._type_list.count(2))))
        print("Number of non-patient negative samples: {}".format(str(self._type_list.count(0))))        
        print("Number of total negative samples: {}".format(str(self._type_list.count(0) + self._type_list.count(2))))        
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")


    def __len__(self):
        return len(self._data_list)


    def __getitem__(self, index):
        return self._data_list[index]