# GRU-D preprocess
import argparse
import pdb
import pandas as pd
import os
import pickle
import numpy as np
from builder.utils.utils import *
from builder.utils.process_util import *
from tqdm import tqdm
import random
import re

def extract_mean(data_list):
    data = np.concatenate(data_list, axis=0)
    mean = np.nanmean(data, axis=0)

    return mean

def cal_time_delta(mask):
    delta = [np.zeros(mask.shape[1])]
    for i in range(1, mask.shape[0]):
        delta.append(np.where(mask[i]==0, delta[i-1], np.zeros(mask.shape[1])) + np.ones(mask.shape[1]))
    delta = np.stack(delta, axis=0)

    return delta

def carry_forward(np_arr, feature_means):
    df = pd.DataFrame(np_arr)
    df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')
    data = df.to_numpy()
    
    for idx, mean in enumerate(feature_means):
        if np.isnan(data[:, idx]).all():
            data[:, idx] = np.array([mean]*len(data))
    
    for idx, mean in enumerate(feature_means):
            data[:, idx] = np.nan_to_num(data[:, idx], nan=mean)

    return data

def train_test_data_split_severancevc(full_data_path_list):
    data_path_list = search_walk({'path': full_data_path_list, 'extension': ".pkl"})
    random.shuffle(data_path_list)
    data_path_list = list([i.split("/")[-1] for i in data_path_list])

    cxr_data = []
    no_cxr_data = []
    for idx, data_path in enumerate(data_path_list):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if data['CXR'] == None:
            no_cxr_data.append(data_path_list[idx])
        else:
            cxr_data.append(data_path_list[idx])

    print("no_cxr_data: ", len(no_cxr_data))
    print("cxr_data: ", len(cxr_data))

    no_cxr_data_size = len(no_cxr_data)
    cxr_data_size = len(cxr_data)

    cxr_test_size = int(cxr_data_size * 0.1)
    cxr_test_dir = list(cxr_data[:cxr_test_size])
    cxr_data_path_list = list(cxr_data[cxr_test_size:])

    data_path_list = list(cxr_data_path_list + no_cxr_data)
    test_dir = [i + "\n" for i in cxr_test_dir]
    data_path_list = [i + "\n" for i in data_path_list]

    print("train_data: ", len(data_path_list))
    print("test_data: ", len(test_dir))
    with open("./data/mimic4_vital_lab_cxr_testlist.txt", "w") as file:
        file.writelines(test_dir)
    with open("./data/mimic4_vital_lab_cxr_trainlist.txt", "w") as file:
        file.writelines(data_path_list)
    exit(1)

def train_test_data_split_severance_vc(base_path):
    file_list = search_walk({'path': base_path, 'extension': ".pkl"})

    patient_dict = {}
    for pkl in file_list:
        pat_id = pkl.split("/")[-1].split("_")[0]
        if pat_id not in patient_dict:
            patient_dict[pat_id] = []
        patient_dict[pat_id].append(pkl)

    pat_num = len(patient_dict)
    print("Number of patients are: ", pat_num)
    print("Number of files are: ", len(file_list))
    patient_list = list(patient_dict.items())
    random.shuffle(patient_list)

    # For now, let's just get random 10% of the dataset as test dataset
    test_size = int(len(patient_list) * 0.1)
    train_dirs_pat = list(patient_list[test_size:])
    test_dirs_pat = list(patient_list[:test_size])
    
    train_dirs = [i[1] for i in train_dirs_pat]
    train_dirs = [item for sublist in train_dirs for item in sublist]
    test_dirs = [i[1] for i in test_dirs_pat]
    test_dirs = [item for sublist in test_dirs for item in sublist]

    print("1_1. Training data number of patients: ", len(train_dirs_pat))
    print("1_2. Training files number: ", len(train_dirs))

    print("2_1. Test data number of patients: ", len(test_dirs_pat))
    print("2_2. Test files number: ", len(test_dirs))

    return train_dirs, test_dirs

def multi_process_function(data_path):
    if not os.path.exists(data_path):
        # raise RuntimeError(f'There is no {data_path}')
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)            

    data['age'] = (int(data['age']) - 18) / (90 - 18)
    feature = data['inputs']
    
    for i in range(feature.shape[1]):
        # if np.isnan(feature[:,-1]).all():
        if all(v is None for v in feature[:,-1]):            
            feature = np.delete(feature, -1, 1)
        else:
            break
        
    if feature.shape[1] == 0:
        return

    feature = np.array(feature, dtype=np.float).T
    mask = (~np.isnan(feature[:, :])).astype(float)
    feature = carry_forward(feature[:, :], FEATURE_MEANS)
    delta = cal_time_delta(mask)
    
    final_dict = {}

    final_dict['pat_id'] = data['pat_id']
    final_dict['chid'] = data['hadm_id']
    final_dict['edstay'] = data['edstay']

    # final_dict['CXR'] = data['CXR']
    
    final_dict['admission_time'] = data['admission_time']
    final_dict['discharge_time'] = data['discharge_time']
    
    # targets
    final_dict['death_time'] = data['death_time']
    final_dict['death_yn'] = data['death_yn']

    # final_dict['intubation_inputs'] = data['intubation_inputs']
    final_dict['intubation_time'] = data['intubation_time']
    final_dict['intubation_yn'] = data['intubation_yn']
    
    # final_dict['vasso_inputs'] = data['vasso_inputs']
    final_dict['vasso_time'] = data['vasso_time']
    final_dict['vasso_yn'] = data['vasso_yn']
    
    # final_dict['cpr_inputs'] = data['cpr_inputs']
    final_dict['cpr_time'] = data['cpr_time']
    final_dict['cpr_yn'] = data['cpr_yn']

    # input data
    final_dict['traige_vitalsigns'] = data['traige_vitalsigns']

    final_dict['age'] = data['age']
    final_dict['gender'] = data['gender']
    final_dict['data'] = feature
    final_dict['mask'] = mask
    final_dict['delta'] = delta
    final_dict['feature_order'] =  [FEATURE_DICT[i] for i in data['feature_order']]
    final_dict['mean'] = FEATURE_MEANS
    
    # final_dict['cxr_yn'] = data['cxr_yn']
    # final_dict['cxr_input'] = data['cxr_path']
    # final_dict['txt_input'] = data['txt_input']

    cxr_inputs = data['cxr_path']
    txt_inputs = data['txt_input']

    final_dict['window_first_idx_hr'] = data['window_first_idx_hr']
    final_dict['window_last_idx_hr'] = data['window_last_idx_hr']
    # final_dict['ed_last_hours'] = data['ed_last_hours']

    new_cxr_inputs = []
    if cxr_inputs is not None:
        for cxr_input in cxr_inputs:
            time_len = cxr_input[0]
            date = cxr_input[1]
            path = cxr_input[2]
            new_cxr_inputs.append([time_len - data['window_first_idx_hr'], path])
            
    new_txt_inputs = []
    if txt_inputs is not None:
        if len(txt_inputs) == 1:
            if len(txt_inputs[0]) == 1:
                txt = txt_inputs[0][0]
                new_txt_inputs.append([None, txt.lower()])

            else:
                print("check_1 txt_inputs: ", txt_inputs)
                print("check_1 file name: ", data_path.split("/")[-1])
        else:
            new_txt_inputs.append([None, txt_inputs.lower()])

    if len(new_txt_inputs) != 0:
        print("before: ", new_txt_inputs)

        exclusion_txt = ['"', ':', '>', '=', '&', '-', '+', ';', "'", '_', '^', '?', '\\', '(', '.']
        new_filtered_txt_inputs = " ".join([i[1] for i in new_txt_inputs])
        for excl in exclusion_txt:
            new_filtered_txt_inputs = new_filtered_txt_inputs.replace(excl, " ")
        new_filtered_txt_inputs = re.sub(' +', ' ', new_filtered_txt_inputs)
        print("after: ", new_filtered_txt_inputs)


    if len(new_cxr_inputs) == 0:
        img_exist = 0
    else:
        img_exist = 1
        final_dict['cxr_input'] = new_cxr_inputs

    if len(new_txt_inputs) == 0:
        txt_exist = 0
    else:
        if len(new_filtered_txt_inputs) > 0 and len(new_filtered_txt_inputs) != " ":
            txt_exist = 1
            final_dict['txt_input'] = new_filtered_txt_inputs
        else:
            txt_exist = 0

    # if data_type == "train":
    final_dict['feature_mins'] = np.min(feature, axis = 0)
    final_dict['feature_maxs'] = np.max(feature, axis = 0)
    
    file_name = str(data_path.split("/")[-1]).split(".")[0]

    file_name = file_name + "_txt" + str(txt_exist) + "_img" + str(img_exist) + ".pkl"
    # print(final_dict)
    # exit(1)
    with open(f'{FINAL_OUTPUT_DIR}/{file_name}', 'wb') as f:
        pickle.dump(final_dict, f)


def main(args):
    base_path = args.base_path
    extracted_feature_path = base_path + '/extract-feature/MIMICED/extract_feature'
    output_dir = base_path + "/training_data_0706/mimic_ed/"
    train_data_list, test_data_list = train_test_data_split_severance_vc(extracted_feature_path)

    global FINAL_OUTPUT_DIR
    global FEATURE_MEANS
    global FEATURE_DICT
    FEATURE_DICT = {'heartrate': 'HR', 'resprate': 'RR', 'temperature': 'BT', 'sbp': 'SBP', 'dbp': 'DBP', 'o2sat': 'Sat', 'GCS': 'GCS', 'HEMATOCRIT': 'Hematocrit', 'PLATELET': 'PLT', 'WBC': 'WBC', 'BILIRUBIN': 'Bilirubin', 'pH': 'pH', 'HCO3': 'HCO3', 'CREATININE': 'Creatinine', 'LACTATE': 'Lactate', 'POTASSIUM': 'Potassium', 'SODIUM': 'Sodium', 'CRP': 'CRP', None: None}

    data_types = {"train": train_data_list, "test": test_data_list}
    data_mean_list = []
    for data_path in tqdm(train_data_list, desc="Processing mean calculation..."):
        if not os.path.exists(data_path):
            raise RuntimeError(f'There is no {data_path}')
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)  
        if len(data['inputs']) == 0:
            continue     
        data_mean_list.append(np.array(data['inputs'], dtype=float).T)
    
    FEATURE_MEANS = extract_mean(data_mean_list)

    for data_type in tqdm(data_types, desc="Processing 2nd step preprocessor"):
        data_path_list = data_types[data_type]
        FINAL_OUTPUT_DIR = output_dir + data_type

        if not os.path.exists(FINAL_OUTPUT_DIR):
            os.makedirs(FINAL_OUTPUT_DIR)
            
        run_multi_process(multi_process_function, data_path_list)

    # # ##
    # data_path_list = search_walk({'path': extracted_feature_path, 'extension': ".pkl"})
    # for i in data_path_list:
    #     multi_process_function(i)
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-od', type=str, default=None)
    parser.add_argument('--base-path', type=str, default='/nfs/thena/shared/multi_modal')
    args = parser.parse_args()
    
    main(args)