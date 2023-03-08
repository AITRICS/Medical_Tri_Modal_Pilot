import random
import itertools
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from control.config import args
from itertools import groupby

import torch

from builder.utils.utils import *
from builder.data.collate_fn import *

VITALSIGN_LABTEST = ['HR', 'RR', 'BT', 'SBP', 'DBP', 'Sat', 'GCS', 'Hematocrit', 'PLT', 'WBC', 'Bilirubin', 'pH', 'HCO3', 'Creatinine', 'Lactate', 'Potassium', 'Sodium', 'CRP']

# CUDA_VISIBLE_DEVICES=7 python ./2_train.py --project-name test --model gru_d --optim adam --epoch 75 --batch-size 16 --cross-fold-val True --lr-scheduler Single --input-types vslt --output-type mortality --predict-type binary --modality-inclusion fullmodal
class Mortality_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for Mortality Prediction...")
        self._data_list = []
        self.feature_means = []
        self.train_min = []
        self.train_max = []

        self.start_indices = []
        self.end_indices = []
        self._type_list = []
        lengths = []

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if args.input_types == "txt":
                    if "txt1_" not in file_name:
                        continue
                elif args.input_types == "img":
                    if "_img1" not in file_name:
                        continue
            """

            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
            
            seq_length = data_info['data'].shape[0]
            # Since for sequence regression, we need 3 data points in the features as well as 3 data points in the target sequence,
            # We require at least 6 data points in the data. For binary classification problems, this number is 4. (3 feature, 1 target)
            if ("sequence" in args.trainer and seq_length < 6) or seq_length < 3:
                continue
            
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.vitalsign_labtest]):
                    continue
            
            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])

                if data_info['death_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)
            else:
                lengths.append(seq_length)
                if data_info['death_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)
            self._data_list.append(pkl_path)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        else:
            self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]
        
        print("Number of positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples for training: {}".format(str(self._type_list.count(0))))        
        
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

class CPR_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for CPR Prediction...")
        self._data_list = []
        self.train_max = []
        self.train_min = []
        self.feature_means = []
        self._type_list = []

        lengths = []
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if args.input_types == "txt":
                    if "txt1_" not in file_name:
                        continue
                elif args.input_types == "img":
                    if "_img1" not in file_name:
                        continue
            """

            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                seq_length = data_info['data'].shape[0]

            if "sequence" in args.trainer and (len(data_info['data']) < 6):
                continue
            if len(data_info['data']) < 4:
                continue
            
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.vitalsign_labtest]):
                    continue

            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])

                if data_info['cpr_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)
            else:
                lengths.append(seq_length)
                if data_info['cpr_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)

            self._data_list.append(pkl_path)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        else:
            self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]
        
        print("Number of positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples for training: {}".format(str(self._type_list.count(0))))        

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

class Vasso_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for CPR Prediction...")
        self._data_list = []
        self.train_max = []
        self.train_min = []
        self.feature_means = []
        self._type_list = []

        lengths = []
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if args.input_types == "txt":
                    if "txt1_" not in file_name:
                        continue
                elif args.input_types == "img":
                    if "_img1" not in file_name:
                        continue
            """
                    
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                seq_length = data_info['data'].shape[0]

            if "sequence" in args.trainer and (len(data_info['data']) < 6):
                continue
            if len(data_info['data']) < 4:
                continue
            
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.vitalsign_labtest]):
                    continue

            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])

                if data_info['vasso_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)
            else:
                lengths.append(seq_length)
                if data_info['vasso_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)

            self._data_list.append(pkl_path)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        else:
            self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]

        print("Number of positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples for training: {}".format(str(self._type_list.count(0))))        

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


class Intubation_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for CPR Prediction...")
        self._data_list = []
        self.train_max = []
        self.train_min = []
        self.feature_means = []
        self._type_list = []

        lengths = []
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
                if args.input_types == "txt":
                    if "txt1_" not in file_name:
                        continue
                elif args.input_types == "img":
                    if "_img1" not in file_name:
                        continue
            """
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                seq_length = data_info['data'].shape[0]

            if "sequence" in args.trainer and (len(data_info['data']) < 6):
                continue
            if len(data_info['data']) < 4:
                continue
            
            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.vitalsign_labtest]):
                    continue

            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])

                if data_info['intubation_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)
            else:
                lengths.append(seq_length)
                if data_info['intubation_yn'] == 0:
                    self._type_list.append(0)
                else:
                    self._type_list.append(1)

            self._data_list.append(pkl_path)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        else:
            self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]

        print("Number of positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples for training: {}".format(str(self._type_list.count(0))))        

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

class Transfer_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for Transfer Prediction...")
        self._data_list = []
        self.train_max = []
        self.train_min = []
        self.feature_means = []

        total_type_list= []
        self._type_list = []

        lengths = []
        full_modal_samples_count = 0
        incomplete_modal_samples_count = 0
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                seq_length = data_info['data'].shape[0]
                if seq_length < args.window_size:
                    continue

            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            
                if data_info['death_yn'] == 0:
                    self._type_list.append(0)
                    total_type_list.append(0)
                else:
                    seq_inputs = data_info['data']
                    max_start_index = seq_inputs.shape[0] - args.window_size
                    max_start_d_index = round(data_info['death_time'] - args.window_size)
                    min_start_index = round(max_start_d_index - args.prediction_after)

                    if min_start_index < 0:
                        min_start_index = 0 
            
                    if max_start_index < min_start_index:
                        target = 0
                        min_start_index = 0 
                    else:
                        target = 1
                    self._type_list.append(target)
                    total_type_list.append(1)
            else:
                lengths.append(seq_length)
                # if data_info['Death_YN'] == 0:
                #     self._type_list.append(0)
                # else:
                #     self._type_list.append(1)
                if data_info['death_yn'] == 0:
                    self._type_list.append(0)
                    total_type_list.append(0)
                else:
                    seq_inputs = data_info['data']
                    max_start_index = seq_inputs.shape[0] - args.window_size
                    max_start_d_index = round(data_info['death_time'] - args.window_size)
                    min_start_index = round(max_start_d_index - args.prediction_after)

                    if min_start_index < 0:
                        min_start_index = 0 
            
                    if max_start_index < min_start_index:
                        target = 0
                        min_start_index = 0 
                    else:
                        target = 1
                    self._type_list.append(target)
                    total_type_list.append(1)

            self._data_list.append(pkl_path)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        else:
            self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]
        
        print("Number of long-term positive samples: {}".format(str(total_type_list.count(1))))
        print("Number of long-term negative samples: {}".format(str(total_type_list.count(0))))
        print("Number of positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples for training: {}".format(str(self._type_list.count(0))))        

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

class Multitask_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="dataset"):
        print("Preaparing Dataset for Transfer Prediction...")
        self._data_list = []
        self.train_max = []
        self.train_min = []
        self.feature_means = []
        self._type_list = []

        lengths = []
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            file_name = pkl_path.split("/")[-1]
            if args.modality_inclusion == "fullmodal":
                if "img1" not in file_name:
                    continue
            """
            else:
            
                if args.input_types == "txt":
                    if "txt1_" not in file_name:
                        continue
                elif args.input_types == "img":
                    if "_img1" not in file_name:
                        continue
            """
                    
            with open(pkl_path, 'rb') as _f:
                data_info = pkl.load(_f)
                seq_length = data_info['data'].shape[0]

            if ("sequence" in args.trainer and len(data_info['data']) < 6):
                continue

            if "vslt" in args.input_types:
                if not all([True if i in data_info['feature_order'] else False for i in args.vitalsign_labtest]):
                    continue

            if "train" in data_type:
                self.train_min.append(data_info['feature_mins'])
                self.train_max.append(data_info['feature_maxs'])
            else:
                lengths.append(seq_length)
            target = int(data_info['death_yn'] or data_info['cpr_yn'] or data_info['intubation_yn'] or data_info['vasso_yn'])
            self._type_list.append(target)

            self._data_list.append(pkl_path)

        self.feature_means = list(data_info['mean'])

        if "train" in data_type:
            self.train_min = np.min(np.array(self.train_min), axis=0)
            self.train_max = np.max(np.array(self.train_max), axis=0)
            
        else:
            self._data_list = [x for _, x in sorted(zip(lengths, self._data_list))]

        print("Number of positive samples for training: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples for training: {}".format(str(self._type_list.count(0))))        

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