# GRU-D preprocess
import os
import re
import pdb
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from builder.utils.utils import *
from builder.utils.process_util import *

##################################################
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', '-od', type=str, default=None)
parser.add_argument('--base_dir', type=str, default='/nfs/thena/shared/multi_modal')
parser.add_argument('--max-input-size', type=int, default=168)  # take 7 days' records (maximum)
ARG = parser.parse_args()
##################################################
FEATURE_LIST = [
    'PULSE', 'RESP', 'TEMP', 'SBP', 'DBP', 'SpO2', 'GCS',
    'HEMATOCRIT', 'PLATELET', 'WBC', 'BILIRUBIN', 'pH', 'HCO3', 
    'CREATININE', 'LACTATE', 'POTASSIUM', 'SODIUM', 'CRP',
]
FEATURE_DICT = {
    'PULSE'     : 'HR', 
    'RESP'      : 'RR', 
    'TEMP'      : 'BT', 
    'SBP'       : 'SBP', 
    'DBP'       : 'DBP', 
    'SpO2'      : 'Sat', 
    'GCS'       : 'GCS', 
    'HEMATOCRIT': 'Hematocrit', 
    'PLATELET'  : 'PLT', 
    'WBC'       : 'WBC', 
    'BILIRUBIN' : 'Bilirubin', 
    'pH'        : 'pH', 
    'HCO3'      : 'HCO3', 
    'CREATININE': 'Creatinine', 
    'LACTATE'   : 'Lactate', 
    'POTASSIUM' : 'Potassium', 
    'SODIUM'    : 'Sodium', 
    'CRP'       : 'CRP', 
    None        : None
}
FEATURE_MEAN = { 
    'PULSE'     : 85.93695802, 
    'RESP'      : 20.10544135, 
    'TEMP'      : 36.97378611, 
    'SBP'       : 120.00165406, 
    'DBP'       : 62.85878326, 
    'SpO2'      : 96.7560417, 
    'GCS'       : 14.58784295, 
    'HEMATOCRIT': 29.44163972, 
    'PLATELET'  : 200.15499694, 
    'WBC'       : 12.11825286, 
    'BILIRUBIN' : 3.79762327, 
    'pH'        : 7.37816261, 
    'HCO3'      : 24.38824869, 
    'CREATININE': 1.5577265, 
    'LACTATE'   : 2.51239096, 
    'POTASSIUM' : 4.12411448, 
    'SODIUM'    : 138.91951009, 
    'CRP'       : 88.96706267,
} # hard-coded mean values...
##################################################

def set_dir():
    ARG.feature_dir   = os.path.join(ARG.base_dir, '/extract-feature/MIMIC/extract_feature')
    ARG.output_dir    = os.path.join(ARG.base_dir, '/training_data_0725/mimic_icu_size{}/'.format(str(ARG.max_input_size)))
    ARG.splitted_dir  = os.path.join(ARG.base_dir, '/splitted_data/mimicicu')
    ARG.train_dir     = os.path.join(ARG.splitted_dir, 'train')
    ARG.test_dir      = os.path.join(ARG.splitted_dir, 'test')


def train_test_data_split(test_ratio):
    file_list = search_walk({'path': ARG.base_dir, 'extension': ".pkl"})
    patient_dict = {}   # {pat_id : list(pkl_path)}
    for pkl in file_list:
        pat_id = pkl.split("/")[-1].split("_")[0]
        patient_dict[pat_id] = patient_dict.get(pat_id, list())
        patient_dict[pat_id].append(pkl)

    pat_num = len(patient_dict)
    print("Number of patients are: ", pat_num)
    print("Number of files are: ", len(file_list))
    patient_list = list(patient_dict.items())   # list of (pat_id, list(pkl_path))
    random.shuffle(patient_list)

    # For now, let's just get random ${test_ratio} of the dataset as test dataset
    test_size       = int(len(patient_list) * test_ratio)
    train_pat_list  = list(patient_list[test_size:])
    test_pat_list   = list(patient_list[:test_size])
    
    train_data_list = [i[1] for i in train_pat_list]
    train_data_list = [item for sublist in train_data_list for item in sublist] # flatten nested list
    test_data_list  = [i[1] for i in test_pat_list]
    test_data_list  = [item for sublist in test_data_list for item in sublist]  # flatten nested list

    print("1_1. Number of patients in Train dataset: ", len(train_pat_list))
    print("1_2. Number of pkl files in Train dataset: ", len(train_data_list))

    print("2_1. Test data number of patients: ", len(test_pat_list))
    print("2_2. Test files number: ", len(test_data_list))

    return train_data_list, test_data_list


def copy_directory_files(path):
    os.system(f"cp -r {path} {ARG.split_path}")


def cal_feature_mean(data_list):
    data = np.concatenate(data_list, axis=0)
    mean = np.nanmean(data, axis=0)
    for i,feature in enumerate(FEATURE_LIST):
        FEATURE_MEAN[feature] = mean[i]


def carry_forward(np_arr, feature_mean):
    df = pd.DataFrame(np_arr, columns=FEATURE_LIST)
    df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')
    df = df.fillna(value=feature_mean)
    data = df.to_numpy()
    
    # for i, m in enumerate(feature_means):
    #     if np.isnan(data[:, i]).all():
    #         data[:, i] = np.array([m]*len(data))

    # for idx, mean in enumerate(feature_means):
    #     data[:, idx] = np.nan_to_num(data[:, idx], nan=mean)

    return data


def cal_time_delta(mask):
    # mask: (SEQ_LENGTH, NUM_FEATURES)
    delta = [np.zeros(mask.shape[1])]   # list((NUM_FEATURES,))
    for i in range(1, mask.shape[0]):   # 1 ~ SEQ_LENGTH
        delta.append(np.where(mask[i] == 0, delta[i-1], np.zeros(mask.shape[1])) +\
                     np.ones(mask.shape[1]))
    delta = np.stack(delta, axis=0)
    return delta


def preprocess_input_data(data_path):
    # check whether pkl file exists
    if not os.path.exists(data_path):
        return

    # load pkl data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)            

    # normalize age / Q: max(age) = 90? 
    data['age'] = (int(data['age']) - 18) / (90 - 18)
    feature = data['inputs'] # shape: (NUM_FEATURES, SEQ_LENGTH)

    # check from the last data point &
    # delete data point where all features are missing
    for i in range(feature.shape[1]):
        if all(v is None for v in feature[:,-1]):
            feature = np.delete(feature, -1, axis=1)
        else:
            break
    
    # exclude: no data point OR too many data point
    # too many = more than 60 days
    if feature.shape[1] == 0 or feature.shape[1] > 1440:
        return
        
    # 아래 6가지 vital-sign이 각각 initial time 부터 어느 기간만큼 missing 되었는지 검사
    initZeroLens = []
    # gcs and other lab-test values will be replace by 정상값 if not exist... no slice for them
    # 0: 'PULSE', 1: 'RESP', 2: 'TEMP', 3: 'SBP', 4: 'DBP', 5: 'Sat'
    for i in range(6):
        # exclude: at least one of vital sign has no valid entry
        if all(feature[i,:] == None):
            return
        # first data point where feature is missing
        initZeroLens.append(np.equal(feature[i,:],None).argmin())
    
    # for new initial vital-sign starting time:
    # V/S feature 중에서 가장 마지막 missing data point를 가진 것을 기준으로 함
    initZeroLens = [i for i in initZeroLens if i < feature.shape[1]]
    initShiftLen = max(initZeroLens)
    
    # 각 feature별 initial-missing-time steps 계산 (dataset.py에서 carry-backward용)
    allInitZeroLens = []
    for i in range(feature.shape[0]):
        # no valid entry at all data points 
        if all(feature[i,:] == None):
            allInitZeroLens.append(0)
        # first data point where feature is missing 
        else:
            allInitZeroLens.append(np.equal(feature[i,:],None).argmin())
            
    # check V/S features are missing at one data point
    # whether at least 5 features exist --> vsMissSteps
    feature      = np.array(feature, dtype=np.float).T   # shape: (SEQ_LENGTH, NUM_FEATURES)
    mask         = (~np.isnan(feature[:,:])).astype(float)
    vsMissSteps  = np.array([1 if sum(mask[idx,:6]) >= 5 else 0 for idx in range(mask.shape[0])])   # shape: (SEQ_LENGTH,)
    
    # feature missing - carry forward & mean imputation
    # feature mask - 
    feature      = carry_forward(feature[:,:], FEATURE_MEAN)
    delta        = cal_time_delta(mask)
    
    # shift data - from the last V/S missing data point 
    feature      = feature[initShiftLen:,:]  
    mask         = mask[initShiftLen:,:]
    delta        = delta[initShiftLen:,:]
    vsMissSteps  = vsMissSteps[initShiftLen:]
    
    # update start index
    initStartIdx = np.array(allInitZeroLens) - initShiftLen
    initStartIdx[initStartIdx<0] = 0
    
    # exclude: invalid feature array OR too short data points
    if feature.shape[0] is None or feature.shape[0] < 3:
        return
    
    # procedure: check possible window sizes for prediction
    # moving pred_index from 2 to SEQ_LEN-1,
    # count all data points where at least 5 V/S features exist 
    # in range of (0, pred_idx), (1, pred_idx), ... iteratively
    possibleWinSizes = dict()
    # for pred_idx in range(2, feature.shape[0]):    # pred_idx: 2,3,...,SEQ_LENGTH-1
    for pred_idx in range(23, feature.shape[0]):    # pred_idx: 2,3,...,SEQ_LENGTH-1
        if pred_idx < 23:
            dict_items = [i for i in range(1, pred_idx+2) if sum(vsMissSteps[pred_idx-i:pred_idx]) > (pred_idx+2)//3]
        else: 
            dict_items = [i for i in range(1, ARG.max_input_size+1) if sum(vsMissSteps[pred_idx-i:pred_idx]) > i//3]
        
        if len(dict_items) > 0:
            possibleWinSizes[pred_idx] = dict_items
    
    # possibleSizes_Dict = dict()
    # for idx, w_Idx in enumerate(range(2, feature.shape[0])):
    #     dict_items = possibleSizes[idx]
    #     if len(dict_items) > 0:
    #         possibleSizes_Dict[w_Idx] = dict_items
    
    # define & initialize final_dict: preprocessed data
    final_data = dict(
        pat_id           = data['pat_id'],
        chid             = data['chid'],
        admission_time   = data['admission_time'], 
        discharge_time   = data['discharge_time'], 
        initShiftLen     = initShiftLen,
        possibleWinSizes = possibleWinSizes,
    )
    
    # update targets & time point based on changed initial data point(=initShiftLen)
    # target1: death
    final_data['death_yn']   = data['death_yn']
    final_data['death_time'] = data['death_time']
    if final_data['death_yn'] == 1:
        new_death_time = data['death_time'] - initShiftLen
        # exclude: death time before updated intial data point
        if new_death_time < 0:
            return
        else:
            final_data['death_time'] = new_death_time

    # target2: intubation
    final_data['intubation_yn']   = data['intubation_yn']
    final_data['intubation_time'] = data['intubation_time']
    if final_data['intubation_yn'] == 1:
        new_intubation_time = [(t[0]-initShiftLen, t[1]-initShiftLen) for t in data['intubation_time'] if t[0]-initShiftLen > 0]
        if len(new_intubation_time) > 0:
            final_data['intubation_time'] = new_intubation_time
        else:
            final_data['intubation_yn']   = 0
            final_data['intubation_time'] = None
    
    # target3: vasso
    final_data['vasso_yn']   = data['vasso_yn']
    final_data['vasso_time'] = data['vasso_time']
    if final_data['vasso_yn'] == 1:
        new_vasso_time = [(t[0]-initShiftLen, t[1]-initShiftLen) for t in data['vasso_time'] if t[0]-initShiftLen > 0]
        if len(new_vasso_time) > 0:
            final_data['vasso_time'] = new_vasso_time
        else:
            final_data['vasso_yn']   = 0
            final_data['vasso_time'] = None
    
    # target4: cpr
    final_data['cpr_yn']   = data['cpr_yn']
    final_data['cpr_time'] = data['cpr_time']
    if final_data['cpr_yn'] == 1:
        new_cpr_time = [(t[0]-initShiftLen, t[1]-initShiftLen) for t in data['cpr_time'] if t[0]-initShiftLen > 0]
        if len(new_cpr_time) > 0:
            final_data['cpr_time'] = new_cpr_time
        else:
            final_data['cpr_yn']   = 0
            final_data['cpr_time'] = None
    
    # input data
    final_data['age']                 = data['age']
    final_data['gender']              = data['gender']
    final_data['data']                = feature
    final_data['mask']                = mask
    final_data['delta']               = delta
    final_data['vsMissSteps']         = vsMissSteps
    final_data['initStartIdx']        = initStartIdx
    final_data['window_first_idx_hr'] = data['window_first_idx_hr'] + initShiftLen
    final_data['feature_mins']        = np.min(feature, axis = 0)
    final_data['feature_maxs']        = np.max(feature, axis = 0)
    
    # rename feature names
    final_data['feature_order']       = [FEATURE_DICT[i] for i in data['feature_order']] 
    final_data['mean']                = {FEATURE_DICT[key]:val for key,val in FEATURE_MEAN.items()}
    
    # update cxr_inputs
    cxr_inputs = data['cxr_path']
    if cxr_inputs is not None:
        new_cxr_inputs = []
        for cxr_input in cxr_inputs:
            time_len     = cxr_input[0]
            date         = cxr_input[1]
            path         = cxr_input[2]
            new_time_len = time_len - final_data['window_first_idx_hr']
            # check: time validity (include only cxr data after window_first_idx)
            if new_time_len >= 0:
                new_cxr_inputs.append([new_time_len, path])
        # add cxr info in final_data
        if len(new_cxr_inputs) > 0:
            final_data['cxr_yn']    = 1
            final_data['cxr_input'] = new_cxr_inputs
        else:
            final_data['cxr_yn']    = 0
            final_data['cxr_input'] = None
    else:
        # add cxr info in final_data
        final_data['cxr_yn']    = 0
        final_data['cxr_input'] = None
    
    # update txt_inputs
    txt_inputs = data['txt_input']  # nested list(chiefcomplaints)
    if len(txt_inputs) > 0:
        if len(txt_inputs) == 1:
            if len(txt_inputs[0]) == 1: 
                txt            = txt_inputs[0][0]
                new_txt_inputs = [None, txt.lower()]
                # new_txt_inputs.append([None, txt.lower()])
            else:
                print("1_1 txt_inputs: ", txt_inputs)
                print("1_1 file name:  ", data_path.split("/")[-1])
        else:
            txt            = " ".join([i_t[0] for i_t in txt_inputs])  # seperate words by white blank (" ")
            new_txt_inputs = [None, txt.lower()]
            # new_txt_inputs.append([None, txt.lower()])
    
    # filter txt input
    if len(new_txt_inputs) > 0:
        exclusion_txt           = ['"', ':', '>', '=', '&', '-', '+', ';', "'", '_', '^', '?', '\\', '(', '.']
        new_filtered_txt        = new_txt_inputs[1]
        for excl in exclusion_txt:
            new_filtered_txt    = new_filtered_txt.replace(excl, " ")
        new_filtered_txt        = re.sub(' +', ' ', new_filtered_txt)
        # add txt info in final_data
        final_data['txt_yn']    = 1
        final_data['txt_input'] = [new_filtered_txt]
    else:
        # add txt info in final_data
        final_data['txt_yn']    = 0
        final_data['txt_input'] = None 
    
    # save .pkl file
    file_name = str(data_path.split("/")[-1]).split(".")[0]
    file_name = file_name + "_txt" + str(final_data['txt_yn']) + "_img" + str(final_data['cxr_yn']) + ".pkl"
    with open(f'{ARG.final_output_dir}/{file_name}', 'wb') as f:
        pickle.dump(final_data, f)


def main():
    # set directories
    set_dir()

    # check whether splitted_dir exists
    if not os.path.exists(ARG.splitted_dir):
        # split dataset into train / test
        print("Previous split history not exist... split...")
        train_data_list, test_data_list = train_test_data_split(test_ratio=0.1)
        
        # create new directories
        os.makedirs(ARG.splitted_dir)
        os.makedirs(ARG.train_dir)
        os.makedirs(ARG.test_dir)
        
        # copy files to ARG.train_dir
        ARG.split_path = ARG.train_dir
        run_multi_process(copy_directory_files, train_data_list)

        # copy files to ARG.test_dir 
        ARG.split_path = ARG.test_dir
        run_multi_process(copy_directory_files, test_data_list)
    else:
        train_data_list = search_walk({'path': ARG.train_dir, 'extension': ".pkl"})
        test_data_list  = search_walk({'path': ARG.test_dir,  'extension': ".pkl"})

    data_types = {"train": train_data_list, "test": test_data_list}
    # # step1: calculate train set mean
    # data_mean_list = []
    # for data_path in tqdm(train_data_list, desc="Processing mean calculation..."):
    #     if not os.path.exists(data_path):
    #         raise RuntimeError(f'There is no {data_path}')
    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)  
    #     if len(data['inputs']) == 0:
    #         continue     
    #     data_mean_list.append(np.array(data['inputs'], dtype=float).T)
    # # calculate & set FEATURE_MEAN
    # cal_feature_mean(data_mean_list)
    # print("FEATURE_MEAN: ", FEATURE_MEAN)

    # step2: missing imputation
    for data_type in tqdm(data_types, desc="Processing 2nd step preprocessor"):
        data_list            = data_types[data_type]
        ARG.final_output_dir = os.path.join(ARG.output_dir, data_type)
        # create final_output_dir 
        if not os.path.exists(ARG.final_output_dir):
            os.makedirs(ARG.final_output_dir)
        # data preprocess - carry forward, mean imputation...
        run_multi_process(preprocess_input_data, data_list)
        
    ## 디버깅 용
    # for i in train_data_list:
    #     multi_process_function(i)

if __name__=='__main__':
    main()
