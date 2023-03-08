# coding: utf-8

import os
import argparse
import pickle
import re
import numpy as np
import math
import csv
from datetime import datetime, timedelta

from common.util.datetime_util import *
from common.util.process_util import *
from common.feature_table.feature_table_mimic import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir'         , type=str)
parser.add_argument('--in_dir'          , type=str)
parser.add_argument('--image_file'      , type=str, default="/mnt/aitrics_ext/ext01/MedicalAI/ImageData/public/MIMIC_CXR/physionet.org/files/mimic-cxr/2.0.0/")
parser.add_argument('--triage_file'      , type=str, default="/mnt/aitrics_ext/ext01/MedicalAI/VC_data/original/public/MIMIC-IV_ED/physionet.org/files/mimic-iv-ed/1.0/ed/triage.csv")
parser.add_argument('--edstay_file'      , type=str, default="/mnt/aitrics_ext/ext01/MedicalAI/VC_data/original/public/MIMIC-IV_ED/physionet.org/files/mimic-iv-ed/1.0/ed/edstays.csv")
parser.add_argument('--mimic_reform_path'      , type=str, default="/mnt/aitrics_ext/ext01/shared/multi_modal/reform-rawdata/MIMIC/merge_by_unitNo/")
ARG = parser.parse_args()

VASSO_ITEM_IDS = ['221906', '221662', '221653', '221289']
INTUBATION_CODES = ['223059', '224385', '225307', '225308', '225468', '225477', '226188', '227194'] # 226188-시작지점, 227194-종결지점
CPR_ID = '225466'    

def get_patids():
    return os.listdir(ARG.in_dir)

def datetime_to_hours(time):
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    result = (year * 8760) + (month * 730) + (day * 24) + hour + (minute / float(60))
    
    return result

def read_patient_file(features, path, stay_id):
    edstay_file = os.path.join(path, stay_id, 'edstays.pkl')
    if not os.path.exists(edstay_file):
        return features, False, 0

    with open(edstay_file, 'rb') as f:
        admin_info = pickle.load(f)[0]

    PATIENTS = edstay_file
    # subject_id = admin_info['subject_id']
    hadm_id = admin_info['hadm_id']
    
    if len(hadm_id) == 0:
        return features, False, 0
    
    admin_dt = str2datetime(admin_info['intime'])
    disch_dt = str2datetime(admin_info['outtime'])
    
    length = datetime_to_hours(disch_dt) - datetime_to_hours(admin_dt)

    if admin_dt != NOT_CONVERTED:
        features['hadm_id'] = hadm_id
        features['admission_time'] = admin_dt
        features['discharge_time'] = disch_dt
        flag = True
    else:
        flag = False

    return features, flag, length

def read_triage_file(features, path, stay_id):
    in_file_base = 'triage.pkl'
    features_to_use = ['heartrate', 'resprate', 'temperature', 'sbp', 'dbp', 'o2sat']
    in_file = os.path.join(path, stay_id, in_file_base)
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)[0]

    features["traige_vitalsigns"] = []
    for feature in features_to_use:
        features["traige_vitalsigns"].append(d[feature])

    features["txt_input"] = d['chiefcomplaint']

    return features

def read_vitalsign_file(features, path, stay_id, counts):
    in_file_bases = [
        'vitalsign.pkl',
    ]
    features_to_use = ['heartrate', 'resprate', 'temperature', 'sbp', 'dbp', 'o2sat']
    for in_file_base in in_file_bases:
        in_file = os.path.join(path, stay_id, in_file_base)
        if not os.path.exists(in_file):
            continue

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            feature_dt = str2datetime(feature_item['charttime'])
            if feature_dt == NOT_CONVERTED:
                continue

            for feature_name in features_to_use:
                feature_value = feature_item[feature_name]
                
                if len(feature_value) == 0:
                    continue

                if feature_name == 'temperature':
                    feature_value = (float(feature_value) - 32.0) / 1.8

                features[feature_name] = features.get(feature_name, list())
                features[feature_name].append((feature_value, feature_dt))
    
    return features

def read_labresult_file(features, path, stay_id):
    features_to_use = ['HEMATOCRIT', 'PLATELET', 'WBC', 'BILIRUBIN', 'pH', 'HCO3', 
                        'CREATININE', 'LACTATE', 'POTASSIUM', 'SODIUM', 'CRP'] #, 'CRP'
    
    mimic_labevent_path = ARG.mimic_reform_path + features['pat_id'] + "/" + features['hadm_id'] + "/LABEVENTS.pkl"
    
    if not os.path.exists(mimic_labevent_path):
        return features, True

    with open(mimic_labevent_path, 'rb') as f:
        d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['ITEMID']
            if context_id not in FEATURE_FOR_EXTRACT_FEATURE.LAB:
                continue

            feature_code = FEATURE_FOR_EXTRACT_FEATURE.LAB[context_id]
            
            feature_time = str2datetime(feature_item['CHARTTIME'])
            if feature_time == NOT_CONVERTED:
                continue

            try:
                # feature_value = float(feature_item['NmrcRslt'])
                feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['VALUENUM'])[0])
            except Exception:
                continue

            # nine_hours_from_now = datetime.now() + timedelta(hours=9)
            # print("1 ed discharge time: ", )
            # print("2 feature_time: ", feature_time)
            # exit(1)

            # ER 응급실을 떠난지 24시간안에 measure 된 labtest가 아닐경우 제외
            if feature_time < (features['discharge_time'] + timedelta(hours=24)):
                features[feature_code] = features.get(feature_code, list())
                features[feature_code].append((feature_value, feature_time))

    # current_features = list(features.keys())
    return features, True

def update_image_info(features, path, stay_id):
    if stay_id in ARG.image_dict:
        features["cxr_path"] = list(sorted(ARG.image_dict[stay_id]))
        img_count = 1
    else:
        features["cxr_path"] = None
        img_count = 0
    
    return features, img_count


def hourly_extract(features, path, chid):
    feature_types = ['heartrate', 'resprate', 'temperature', 'sbp', 'dbp', 'o2sat',
                        'GCS', 'HEMATOCRIT', 'PLATELET', 'WBC', 'BILIRUBIN', 'pH', 'HCO3', 
                        'CREATININE', 'LACTATE', 'POTASSIUM', 'SODIUM', 'CRP']
    feature_normal_values = {'HEMATOCRIT':45, 'PLATELET':300, 'WBC':7, 
                        'BILIRUBIN':0.6, 'pH':7.4, 'HCO3':24, 
                        'CREATININE':0.8, 'LACTATE':0.7, 'POTASSIUM':4.2, 'SODIUM':140}

    # 응급실 입실 시간 기준 첫시간과 마지막 시간
    first_time = datetime_to_hours(features['admission_time']) 
    # last_time = datetime_to_hours(features['discharge_time'] + timedelta(hours=24))
    ed_last_time = datetime_to_hours(features['discharge_time'])
    
    first_hours = int(math.ceil(first_time))
    ed_last_hours = int(math.ceil(ed_last_time))

    # ICU 시간 기준 첫시간기준으로 death_time 구함
    if features['death_yn'] == 1:
        death_time = datetime_to_hours(features['death_time'])
        if death_time <= first_hours:
            return features, False

        death_after = death_time - first_hours
        features['death_time'] = death_after
        
        # death time 후에 input window 구간이 있을경우 겹칠경우 window size를 death time시간 바로전 까지로 수정
        if death_time <= ed_last_time:
            ed_last_time = int(math.ceil(death_time)) - 1

    # lab-test를 뽑기 위해 last-time을 응급실 discharge후 24시간후 까지도 포함
    last_hours = int(ed_last_hours + 24)

    if features['cpr_yn'] == 1:
        features['cpr_time'] = sorted(features['cpr_time'])
        cpr_after = [(np.round(datetime_to_hours(str2datetime(i[0])) - first_hours, 4), np.round(datetime_to_hours(str2datetime(i[1])) - first_hours, 4)) for i in features['cpr_time']]
        cpr_after = [(i, q) for i, q in cpr_after if i >= 0 and q >= 0]
        features['cpr_time'] = list(cpr_after) 

    if features['vasso_yn'] == 1:
        features['vasso_time'] = sorted(features['vasso_time'])
        vasso_after = [(np.round(datetime_to_hours(str2datetime(i[0])) - first_hours, 4), np.round(datetime_to_hours(str2datetime(i[1])) - first_hours, 4)) for i in features['vasso_time']]
        vasso_after = [(i, q) for i, q in vasso_after if i >= 0 and q >= 0]
        features['vasso_time'] = vasso_after

    if features['intubation_yn'] == 1:
        features['intubation_time'] = sorted(features['intubation_time'])
        intubation_after = [(np.round(datetime_to_hours(str2datetime(i[0])) - first_hours, 4), np.round(datetime_to_hours(str2datetime(i[1])) - first_hours, 4)) for i in features['intubation_time']]
        intubation_after = [(i, q) for i, q in intubation_after if i >= 0 and q >= 0]
        features['intubation_time'] = intubation_after

    all_features = []
    features_in_order = []
    # ceil 된 er 시간 기준 첫 시간 ~ 마지막 시간 + 24시간 기준으로 
    for i in feature_types:
        if i in features:
            features_in_order.append(i)
            values_list = [q[0] for q in features[i]]
            # time_list = [datetime_to_hours(q[1]) if q[1] != -1 else None for q in features[i]]
            # rd_time_list = [round(datetime_to_hours(q[1])) if q[1] != -1 else None for q in features[i]]
            time_list = [datetime_to_hours(q[1]) for q in features[i]]
            ceil_time_list = [math.ceil(datetime_to_hours(q[1])) for q in features[i]]

            one_feature = []
            # deleted_feature = [None, -1]
            for hr_idx in range(first_hours, last_hours+1):
                d_feature = []
                while hr_idx in ceil_time_list:
                    index = ceil_time_list.index(hr_idx)
                    if len(d_feature) == 0:
                        d_feature = [values_list[index], time_list[index]]
                    else:
                        if abs(d_feature[1] - hr_idx) <= abs(time_list[index] - hr_idx):
                            pass
                        else:
                            d_feature = [values_list[index], time_list[index]]
                    del values_list[index]
                    del time_list[index]
                    del ceil_time_list[index]

                if len(d_feature) == 0:
                    one_feature.append(None)
                else:
                    one_feature.append(d_feature[0])
            all_features.append(one_feature)
        else:
            features_in_order.append(None)
            none_feature = list([None for _ in range(first_hours, last_hours+1)])
            all_features.append(none_feature)

    final_features = np.array(all_features)
    sample_len = final_features.shape[1]
    # cpr_samples = np.array([0] * sample_len)
    # vasso_samples = np.array([0] * sample_len)
    # intubation_samples = np.array([0] * sample_len)

    # if features['cpr_time'] == None:
    #     pass
    # else:
    #     for start, end in features['cpr_time']:
    #         start_hr = int(round(start))
    #         end_hr = int(round(end))
    #         if end_hr < start_hr:
    #             continue
    #         end_hr += 1
    #         if start_hr >= len(cpr_samples) or end_hr >= len(cpr_samples):
    #             continue
    #         cpr_samples[start_hr:end_hr] = 1
    
    # if features['vasso_time'] == None:
    #     pass
    # else:
    #     for start, end in features['vasso_time']:
    #         start_hr = int(round(start))
    #         end_hr = int(round(end))
    #         if end_hr < start_hr:
    #             continue
    #         end_hr += 1
    #         if start_hr >= len(vasso_samples) or end_hr >= len(vasso_samples):
    #             continue
    #         vasso_samples[start_hr:end_hr] = 1
     
    # if features['intubation_time'] == None:
    #     pass
    # else:
    #     for start, end in features['intubation_time']:
    #         start_hr = int(round(start))
    #         end_hr = int(round(end))
    #         if end_hr < start_hr:
    #             continue
    #         end_hr += 1
    #         if start_hr >= len(intubation_samples) or end_hr >= len(intubation_samples):
    #             continue
    #         intubation_samples[start_hr:end_hr] = 1

    all_none_steps = 0
    none_steps_flag = False
    for hour_step in range(final_features.shape[1]):
        if all(v is None for v in final_features[:,hour_step]):
            all_none_steps += 1
            none_steps_flag = True
        else:
            break
            
    if none_steps_flag:
        first_hours += all_none_steps

        if first_hours >= last_hours:
            return features, False
        
        final_features = final_features[:, all_none_steps:]
        if features['death_yn'] == 1:
            features['death_time'] = features['death_time'] - all_none_steps
        if features['cpr_yn'] == 1:
            features['cpr_time'] = [(i-all_none_steps, q-all_none_steps) for i, q in features['cpr_time']]
            # cpr_samples = list(cpr_samples)[all_none_steps:]
        if features['vasso_yn'] == 1:
            features['vasso_time'] = [(i-all_none_steps, q-all_none_steps) for i, q in features['vasso_time']]
            # vasso_samples = list(vasso_samples)[all_none_steps:]
        if features['intubation_yn'] == 1:
            features['intubation_time'] = [(i-all_none_steps, q-all_none_steps) for i, q in features['intubation_time']]
            # intubation_samples = list(intubation_samples)[all_none_steps:]
    # else:
    #     cpr_samples = list(cpr_samples)
    #     vasso_samples = list(vasso_samples)

    for i in feature_types:
        if i in features:
            del features[i]

    # features['cpr_inputs'] = cpr_samples
    # features['vasso_inputs'] = vasso_samples
    # features['intubation_inputs'] = intubation_samples
    features['inputs'] = final_features
    features['feature_order'] = features_in_order

    features['window_first_idx_hr'] = first_hours
    features['window_last_idx_hr'] = last_hours

    return features, True


def extract_targets(features, path, chid):
    procedure_events_file = os.path.join(path, chid, 'PROCEDUREEVENTS.pkl')
    input_events_file = os.path.join(path, chid, 'INPUTEVENTS.pkl')

    mimic_admission_path = ARG.mimic_reform_path + features['pat_id'] + "/" + chid + "/ADMISSIONS.pkl"
    if not os.path.exists(mimic_admission_path):
        return features, False

    patients_path = ARG.mimic_reform_path + features['pat_id'] + '/PATIENTS.pkl'
    pat_info = pickle.load(open(patients_path, 'rb'))

    age_dt = pat_info[0]['ANCHOR_AGE']
    gender = pat_info[0]['GENDER']
    
    if age_dt != NOT_CONVERTED:
        features['age'] = age_dt
        features['gender'] = gender
    else:
        return features, False

    with open(mimic_admission_path, 'rb') as f:
        admin_info = pickle.load(f)[0]

    death_dt = str2datetime(admin_info["DEATHTIME"])

    features['death_time'] = death_dt
    if death_dt == "NOT_CONVERTED":
        features['death_yn'] = 0
    else:
        features['death_yn'] = 1

    if not os.path.exists(procedure_events_file):
        features['cpr_yn'] = 0
        features['cpr_time'] = None
        features['intubation_yn'] = 0
        features['intubation_time'] = None
    else:
        with open(procedure_events_file, 'rb') as f:
            d = pickle.load(f)
            cpr_time_list = [(event['STARTTIME'], event['ENDTIME']) for event in d if event['ITEMID'] == CPR_ID]
            if len(cpr_time_list) == 0:
                features['cpr_yn'] = 0
                features['cpr_time'] = None
            else:
                features['cpr_yn'] = 1
                features['cpr_time'] = cpr_time_list
            
            intubation_time_list = [(event['STARTTIME'], event['ENDTIME']) for event in d if event['ITEMID'] in INTUBATION_CODES]
            if len(intubation_time_list) == 0:
                features['intubation_yn'] = 0
                features['intubation_time'] = None
            else:
                features['intubation_yn'] = 1
                features['intubation_time'] = intubation_time_list

    if not os.path.exists(input_events_file):
        features['vasso_yn'] = 0
        features['vasso_time'] = None
    else:
        with open(input_events_file, 'rb') as f:
            d = pickle.load(f)
            vasso_time_list = [(event['STARTTIME'], event['ENDTIME']) for event in d if event['ITEMID'] in VASSO_ITEM_IDS]
            if len(vasso_time_list) == 0:
                features['vasso_yn'] = 0
                features['vasso_time'] = None
            else:
                features['vasso_yn'] = 1
                features['vasso_time'] = vasso_time_list

    return features, True

def load_image_info():
    edstay_dict = {}
    image_dict = {}

    edstay_f = open(ARG.edstay_file, 'r', encoding='utf-8')
    f = open(ARG.image_file + "CXR_VALID_CHID.csv", 'r', encoding='utf-8')
    
    rdr = csv.reader(f)
    rd_edstay = csv.reader(edstay_f)

    for idx, line in enumerate(rd_edstay):
        if idx == 0:
            continue

        subject_id, hadm_id, stay_id, intime, outtime = line
        
        if hadm_id not in edstay_dict:
           edstay_dict[hadm_id] = stay_id

    edstay_f.close()   

    for idx, line in enumerate(rdr):
        if idx == 0:
            continue

        HADM_ID, STUDY_ID, DICOM_ID, ACQ_TIME, PATH, JPG_PATH, CXR_TYPE = line

        if CXR_TYPE == "AP" or CXR_TYPE == "PA":
            if HADM_ID in edstay_dict:
                edstay_id = edstay_dict[HADM_ID]
                if edstay_id not in image_dict:
                    image_dict[edstay_id] = []
                
                date, time = ACQ_TIME.split(" ")
                yr, month, day = list(map(int, date.split("-")))
                hr, min, sec = list(map(int, time.split(":")))
                time_length = (yr * 8760) + (month * 730) + (day * 24) + hr + (min / float(60)) + (sec / float(3600))

                image_dict[edstay_id].append((time_length, ACQ_TIME, JPG_PATH))

    f.close()    

    ARG.image_dict = image_dict

            
def write_feature(pat_id, chid, features):
    out_file = os.path.join(ARG.out_dir + "/{}".format(pat_id), '{}_{}.pkl'.format(pat_id, chid))
    with open(out_file, 'wb') as f:
        pickle.dump(features, f)

def extract_feature_each(pat_id, counts=None):
    path = ARG.in_dir + "/" + str(pat_id)
    stay_id_list = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ] # 1 or more files can exist

    label_count = {'cpr':0, 'vasso':0, 'intubation':0, 'death':0, 'txt':0}
    length_list = []
    for stay_id in stay_id_list:
        features = {'pat_id': pat_id, 'edstay': stay_id}

        features, flag, stay_l = read_patient_file(features, path, stay_id)

        if flag == False:
            # if no hadm_id --> continue
            continue
        chid = features['hadm_id']

        features = read_vitalsign_file(features, path, stay_id, counts)

        features = read_triage_file(features, path, stay_id)

        features, flag = read_labresult_file(features, path, stay_id)

        features, img_count = update_image_info(features, path, stay_id)

        features, flag = extract_targets(features, path, chid)
        if flag == False:
            # if no death y/n --> continue
            continue

        features, flag = hourly_extract(features, path, chid)
        if flag == False:
            continue

        if not os.path.isdir(ARG.out_dir + "/{}".format(pat_id)):
            os.makedirs(ARG.out_dir + "/{}".format(pat_id))

        # print("features: ", features)
        
        write_feature(pat_id, chid, features)

        # return list(length_list)

def main():
    pat_ids = get_patids()
    load_image_info()

    run_multi_process(extract_feature_each, pat_ids)

    # counts = {'HEMATOCRIT':0, 
    # 'PLATELET':0, 
    # 'WBC':0, 
    # 'BILIRUBIN':0, 
    # 'pH':0, 
    # 'HCO3':0, 
    # 'CREATININE':0, 
    # 'LACTATE':0, 
    # 'POTASSIUM':0, 
    # 'SODIUM':0, 
    # 'CRP':0, 
    # 'PULSE':0, 
    # 'RESP':0, 
    # 'TEMP':0, 
    # 'SBP':0, 
    # 'DBP':0, 
    # 'SpO2':0,
    # 'GCS':0}

    # counts = None
    # for idx, pat_id in enumerate(pat_ids):
    #     print("{} / {}".format(str(idx+1), str(len(pat_ids))))
    #     extract_feature_each(pat_id, counts)
        
        # total_lens = total_lens + lens
    # from matplotlib import pyplot as plt
    # plt.hist(total_lens, bins=1000)
    # plt.show()
    # print("final counts: ", counts)

if __name__ == '__main__':
    main()
