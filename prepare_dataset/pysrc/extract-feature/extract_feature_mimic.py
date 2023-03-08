# coding: utf-8

import os
import argparse
import pickle
import re
import numpy as np
import csv
import math

from common.util.datetime_util import *
from common.util.process_util import *
from common.feature_table.feature_table_mimic import *

##################################################
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir'         , type=str)
parser.add_argument('--in_dir'          , type=str)
parser.add_argument('--image_file'      , type=str, default="/mnt/aitrics_ext/ext01/MedicalAI/ImageData/public/MIMIC_CXR/physionet.org/files/mimic-cxr/2.0.0/")
parser.add_argument('--triage_file'     , type=str, default="/mnt/aitrics_ext/ext01/MedicalAI/VC_data/original/public/MIMIC-IV_ED/physionet.org/files/mimic-iv-ed/1.0/ed/triage.csv")
parser.add_argument('--edstay_file'     , type=str, default="/mnt/aitrics_ext/ext01/MedicalAI/VC_data/original/public/MIMIC-IV_ED/physionet.org/files/mimic-iv-ed/1.0/ed/edstays.csv")
ARG = parser.parse_args()
##################################################
VASSO_ITEM_IDS   = ['221906', '221662', '221653', '221289']
INTUBATION_CODES = ['223059', '224385', '225307', '225308', '225468', 
                    '225477', '226188', '227194'] # 226188-시작지점, 227194-종결지점
CPR_ID           = '225466'    
##################################################


def get_patids():
    return os.listdir(ARG.in_dir)


def load_image_info():
    image_dict = {}
    f = open(ARG.image_file + "CXR_VALID_CHID.csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for idx, line in enumerate(rdr):
        if idx == 0:
            continue
        HADM_ID, STUDY_ID, DICOM_ID, ACQ_TIME, PATH, JPG_PATH, CXR_TYPE = line
        if CXR_TYPE == "AP" or CXR_TYPE == "PA":
            date, time = ACQ_TIME.split(" ")
            yr, month, day = list(map(int, date.split("-")))
            hr, min, sec = list(map(int, time.split(":")))
            time_length = (yr * 8760) + (month * 730) + (day * 24) + hr + (min / float(60)) + (sec / float(3600))
            image_dict[HADM_ID] = image_dict.get(HADM_ID, list())
            image_dict[HADM_ID].append((time_length, ACQ_TIME, JPG_PATH))
    f.close() 
    return image_dict   


def load_txt_info():    
    triage_dict = {}
    edstay_dict = {}
    txt_dict = {}

    triage_f = open(ARG.triage_file, 'r', encoding='utf-8')
    edstay_f = open(ARG.edstay_file, 'r', encoding='utf-8')

    rd_triage = csv.reader(triage_f)
    rd_edstay = csv.reader(edstay_f)

    # triage file --> chief complaint
    for idx, line in enumerate(rd_triage):
        if idx == 0:
            continue
        subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint = line
        triage_dict[stay_id] = triage_dict.get(stay_id, list())
        triage_dict[stay_id].append(chiefcomplaint)
    triage_f.close()    

    # edstay file --> lookup {stay_id: hadm_id}
    for idx, line in enumerate(rd_edstay):
        if idx == 0:
            continue
        subject_id, hadm_id, stay_id, intime, outtime = line
        edstay_dict[stay_id] = edstay_dict.get(stay_id, None)
        edstay_dict[stay_id] = hadm_id
    edstay_f.close()    

    # txt_dict --> {hadm_id: chiefcomplaints}
    for stay_id in triage_dict.keys():
        hadm_id = edstay_dict.get(stay_id, None)
        if not hadm_id:
            continue
        txt_dict[hadm_id] = txt_dict.get(hadm_id, list())
        txt_dict[hadm_id].append(triage_dict[stay_id])
    return txt_dict


def read_patient_file(features, path, chid):
    in_file = os.path.join(path, chid, 'ADMISSIONS.pkl')
    # exclude: no admission info
    if not os.path.exists(in_file):
        return features, False

    with open(in_file, 'rb') as f:
        admin_info = pickle.load(f)[0]

    # admission time, discharge time, age, death, gender 저장
    PATIENTS = pickle.load(open(os.path.join(path, 'PATIENTS.pkl'), 'rb'))[0]   # list(OrderedDict)
    admin_dt = str2datetime(admin_info['ADMITTIME'])
    disch_dt = str2datetime(admin_info['DISCHTIME'])
    death_dt = str2datetime(admin_info["DEATHTIME"])
    age      = int(PATIENTS['ANCHOR_AGE'])
    age_dt   = int(PATIENTS['ANCHOR_YEAR'])
    gender   = PATIENTS['GENDER']

    # exclude: invalid admission/discharge time
    if admin_dt == NOT_CONVERTED or disch_dt == NOT_CONVERTED:
        return features, False

    # set values 
    features['admission_time'] = admin_dt
    features['discharge_time'] = disch_dt
    features['death_time']     = death_dt
    features['death_yn']       = 0 if death_dt == NOT_CONVERTED else 1 
    features['age']            = age + (admin_dt.year - age_dt)
    features['gender']         = gender

    return features, True


def read_observation_file(features, path, chid, counts):
    in_file_bases = [
        'CHARTEVENTS.pkl',
    ]
    features_to_use = [
        'PULSE', 'RESP', 'TEMP', 'SBP', 'DBP', 'SpO2', 'GCS'
    ]
    normal_ranges = {'PULSE': (0, 300), 'RESP': (0,120), 'TEMP': (25,50), 'SBP': (0,300), 'DBP': (0,300), 'SpO2': (0,100), 'GCS': (3,15), 'GCS_EYE': (3,15), 'GCS_VER': (3,15), 'GCS_MOT': (3,15)}

    for in_file_base in in_file_bases:
        # load observation file 
        in_file = os.path.join(path, chid, in_file_base)
        if not os.path.exists(in_file):
            continue

        with open(in_file, 'rb') as f:
            d = pickle.load(f)  # list of records 

        for feature_item in d:
            context_id   = feature_item['ITEMID']
            feature_code = FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION.get(context_id, None) 
            # skip: feature with no interest
            if not feature_code:
                continue

            feature_time = str2datetime(feature_item['CHARTTIME'])
            # skip: invalid feature time
            if feature_time == NOT_CONVERTED:
                continue

            feature_value = feature_item['VALUENUM']

            try:
                feature_value = float(re.findall("\d+\.\d+|\d+", feature_value)[0])
            except Exception:
                continue

            if feature_code == 'TEMP_C':
                feature_code = 'TEMP'
            elif feature_code == 'TEMP_F':
                feature_code = 'TEMP'
                feature_value = (feature_value - 32.0) / 1.8

            normal_range = normal_ranges[feature_code]
            if feature_value < normal_range[0] or feature_value > normal_range[1]:
                continue

            features[feature_code] = features.get(feature_code, list())
            features[feature_code].append((feature_value, feature_time))
    
    # Diff: GCS & GCS_eye/ver/mot can exist in same time
    if 'GCS_EYE' in features.keys() and \
       'GCS_VER' in features.keys() and \
       'GCS_MOT' in features.keys() :
         
        gcs_eye = features['GCS_EYE']
        gcs_ver = features['GCS_VER']
        gcs_mot = features['GCS_MOT']
        
        for eye_value, eye_time in gcs_eye:
            gcs_total = eye_value
            ver_ok = False
            mot_ok = False
        
            # check: GCS_VER
            for ver_value, ver_time in gcs_ver:
                if ver_time == eye_time:
                    gcs_total += ver_value
                    ver_ok = True
                    break
            # check: GCS_MOT
            for mot_value, mot_time in gcs_mot:
                if mot_time == eye_time:
                    gcs_total += mot_value
                    mot_ok = True
                    break
            # sum GCS: eye + ver + mot
            if ver_ok and mot_ok:
                feature_code = 'GCS'
                normal_range = normal_ranges[feature_code]
                if feature_value < normal_range[0] or feature_value > normal_range[1]:
                    continue
                features[feature_code] = features.get(feature_code, list())
                features[feature_code].append((gcs_total, eye_time))

    current_features = list(features.keys())
    current_features = [f for f in current_features if f in features_to_use]
    # exclude: no vital sign
    if len(current_features) == 0:
        return features, False, counts
    # check: feature count
    if counts != None:
        for f in current_features:
            counts[f] += 1

    return features, True, counts


def read_labresult_file(features, path, chid, counts):
    features_to_use = [
        'HEMATOCRIT', 'PLATELET', 'WBC', 'BILIRUBIN', 'pH', 'HCO3', 
        'CREATININE', 'LACTATE', 'POTASSIUM', 'SODIUM', 'CRP'
    ]
    # feature_normal_values = {
    #     'HEMATOCRIT':45, 'PLATELET':300, 'WBC':7, 
    #     'BILIRUBIN':0.6, 'pH':7.4, 'HCO3':24, 
    #     'CREATININE':0.8, 'LACTATE':0.7, 'POTASSIUM':4.2, 'SODIUM':140
    # }
    normal_ranges = {'HEMATOCRIT': (0, 100), 'PLATELET': (0, 1000), 'WBC': (0,100), 'BILIRUBIN': (0,75), 'pH': (0,14), 'HCO3': (0,100), 'CREATININE': (0,20), 'LACTATE': (0,20), 'POTASSIUM': (0,15), 'SODIUM': (0,500), 'CRP': (0,900)}

    # load lab test file
    in_file = os.path.join(path, chid, 'LABEVENTS.pkl')
    if not os.path.exists(in_file):
        return features, counts

    with open(in_file, 'rb') as f:
        d = pickle.load(f)  # list of records

    for feature_item in d:
        context_id = feature_item['ITEMID']
        feature_code = FEATURE_FOR_EXTRACT_FEATURE.LAB.get(context_id, None)
        # skip: feature with no interest
        if not feature_code:
            continue
        
        feature_time = str2datetime(feature_item['CHARTTIME'])
        # skip: invalid feature time
        if feature_time == NOT_CONVERTED:
            continue

        try:
            feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['VALUENUM'])[0])
        except Exception:
            continue

        normal_range = normal_ranges[feature_code]
        if feature_value < normal_range[0] or feature_value > normal_range[1]:
            continue

        features[feature_code] = features.get(feature_code, list())
        features[feature_code].append((feature_value, feature_time))

    current_features = list(features.keys())
    current_features = [f for f in current_features if f in features_to_use]
    # check: feature count
    if counts != None:
        for f in current_features:
            counts[f] += 1

    return features, counts


def update_image_info(features, chid):
    cxr_path = sorted(ARG.img_dict.get(chid, list()))
    if len(cxr_path) == 0:
        features['cxr_path'] = None
        img_count = 0
    else:
        features['cxr_path'] = cxr_path
        img_count = 1
    return features, img_count


def update_txt_info(features, chid):
    txt_input = ARG.txt_dict.get(chid, list())
    if len(txt_input) == 0:
        features['txt_input'] = None
        txt_count = 0
    else:
        features['txt_input'] = txt_input
        txt_count = 1
    return features, txt_count


def extract_targets(features, path, chid):
    procedure_events_file = os.path.join(path, chid, 'PROCEDUREEVENTS.pkl')
    input_events_file     = os.path.join(path, chid, 'INPUTEVENTS.pkl')
    
    if not os.path.exists(procedure_events_file):
        features['cpr_yn'] = 0
        features['cpr_time'] = None
        features['intubation_yn'] = 0
        features['intubation_time'] = None
    else:
        # load cpr/intubation event file
        with open(procedure_events_file, 'rb') as f:
            d = pickle.load(f)

        # check: cpr event (start, end)
        cpr_time_list = [(event['STARTTIME'], event['ENDTIME']) for event in d if event['ITEMID'] == CPR_ID]
        if len(cpr_time_list) == 0:
            features['cpr_yn'] = 0
            features['cpr_time'] = None
        else:
            features['cpr_yn'] = 1
            features['cpr_time'] = cpr_time_list
        # check: intubation event (start, end)
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
        # load vasso event file
        with open(input_events_file, 'rb') as f:
            d = pickle.load(f)

        # check: vasso event (start, end)
        vasso_time_list = [(event['STARTTIME'], event['ENDTIME']) for event in d if event['ITEMID'] in VASSO_ITEM_IDS]
        if len(vasso_time_list) == 0:
            features['vasso_yn'] = 0
            features['vasso_time'] = None
        else:
            features['vasso_yn'] = 1
            features['vasso_time'] = vasso_time_list

    return features


def hourly_extract(features, path):
    feature_types = [
        'PULSE', 'RESP', 'TEMP', 'SBP', 'DBP', 'SpO2', 'GCS',
        'HEMATOCRIT', 'PLATELET', 'WBC', 'BILIRUBIN', 'pH', 'HCO3', 
        'CREATININE', 'LACTATE', 'POTASSIUM', 'SODIUM', 'CRP'
    ]
    # feature_normal_values = {
    #     'HEMATOCRIT':45, 'PLATELET':300, 'WBC':7, 
    #     'BILIRUBIN':0.6, 'pH':7.4, 'HCO3':24, 
    #     'CREATININE':0.8, 'LACTATE':0.7, 'POTASSIUM':4.2, 'SODIUM':140
    # }

    chid = features['chid']
    ICUSTAYS_file = os.path.join(path, chid, 'ICUSTAYS.pkl')
    if not os.path.exists(ICUSTAYS_file):
        return features, False
    with open(ICUSTAYS_file, 'rb') as f:
        ICUSTAYS_info = pickle.load(f)  # can be more than once [MICU --> SICU -->]

    # ICU 시간 기준 첫시간과 마지막 시간
    first_time = min([datetime_to_hours(str2datetime(i['INTIME'])) for i in ICUSTAYS_info])
    last_time = max([datetime_to_hours(str2datetime(i['OUTTIME'])) for i in ICUSTAYS_info])
    
    # first_time = datetime_to_hours(features['admission_time'])
    # last_time = datetime_to_hours(features['discharge_time'])
    
    last_hours = int(math.ceil(last_time))
    first_hours = int(math.ceil(first_time))

    # ICU 시간 기준 첫시간기준으로 death_time 구함
    if features['death_yn'] == 1:
        death_time = datetime_to_hours(features['death_time'])
        # exclude: death time before ICU first time
        if death_time <= first_hours:
            return features, False

        death_after = death_time - first_hours
        features['death_time'] = death_after    # time delta (hr)
        
        # death time 후에 input window 구간이 있어서 겹치는 경우에는
        # window size를 death time 직전까지로 수정
        if death_time <= last_hours:
            last_hours = int(math.ceil(death_time)) - 1

    # 올림 된 ICU first time 기준으로 cpr, vasso, intubation 구간 구함
    if features['cpr_yn'] == 1:
        # calculate time delta (hr) btw ICU first time & cpr start/end time
        # exclude: cpr events before ICU first time
        features['cpr_time'] = sorted(features['cpr_time'])
        cpr_after = [(np.round(datetime_to_hours(str2datetime(i[0])) - first_hours, 4), np.round(datetime_to_hours(str2datetime(i[1])) - first_hours, 4)) for i in features['cpr_time']]
        cpr_after = [(i, q) for i, q in cpr_after if i >= 0 and q >= 0]
        features['cpr_time'] = list(cpr_after) 
        # update label: cpr
        if len(features['cpr_time']) == 0:
            features['cpr_yn']   = 0
            features['cpr_time'] = None

    if features['vasso_yn'] == 1:
        # calculate time delta (hr) btw ICU first time & vasso start/end time
        # exclude: vasso events before ICU first time
        features['vasso_time'] = sorted(features['vasso_time'])
        vasso_after = [(np.round(datetime_to_hours(str2datetime(i[0])) - first_hours, 4), np.round(datetime_to_hours(str2datetime(i[1])) - first_hours, 4)) for i in features['vasso_time']]
        vasso_after = [(i, q) for i, q in vasso_after if i >= 0 and q >= 0]
        features['vasso_time'] = vasso_after
        # update label: vasso
        if len(features['vasso_time']) == 0:
            features['vasso_yn']   = 0
            features['vasso_time'] = None

    if features['intubation_yn'] == 1:
        # calculate time delta (hr) btw ICU first time & intubation start/end time
        # exclude: intubation events before ICU first time
        features['intubation_time'] = sorted(features['intubation_time'])
        intubation_after = [(np.round(datetime_to_hours(str2datetime(i[0])) - first_hours, 4), np.round(datetime_to_hours(str2datetime(i[1])) - first_hours, 4)) for i in features['intubation_time']]
        intubation_after = [(i, q) for i, q in intubation_after if i >= 0 and q >= 0]
        features['intubation_time'] = intubation_after
        # update label: intubation
        if len(features['intubation_time']) == 0:
            features['intubation_yn']   = 0
            features['intubation_time'] = None

    all_features = []
    features_in_order = []
    # ceil 된 ICU 시간 기준 첫 시간 ~ 마지막 시간 기준으로
    # get data array quantized by 1hr
    for f in feature_types:
        # check: at least one data point exists for feature
        if f in features.keys():
            features_in_order.append(f)
            # sort features by time order -- fval: (value, time)
            features[f]    = sorted(features[f], key=lambda x: x[1])
            # get feature values, times (in hr), ceil_times (in hr)
            values_list    = [fval[0] for fval in features[f]]
            ceil_time_list = [math.ceil(datetime_to_hours(fval[1])) for fval in features[f]]
            
            one_feature = []
            for hr_idx in range(first_hours, last_hours+1):
                tmp_feature = None
                # check: whether data point exists in current time point
                # update the latest fval in current time point (in same range n,n+1)
                while hr_idx in ceil_time_list:
                    index = ceil_time_list.index(hr_idx)
                    tmp_feature = values_list[index]
                    del values_list[index]
                    del ceil_time_list[index]
                # save tmp feature in current time point
                one_feature.append(tmp_feature)
            # save feature value list for entire ICU stay
            # including None for the missing time point
            all_features.append(one_feature)
        else:
            features_in_order.append(None)
            none_feature = list([None for _ in range(first_hours, last_hours+1)])
            all_features.append(none_feature)
    
    final_features = np.array(all_features) # shape: (num_features, length_of_stay)

    all_none_steps = 0
    none_steps_flag = False
    # check: first data point among features in ICU stay
    for hour_step in range(final_features.shape[1]):
        # check: whether none of feature has data point
        if all(v is None for v in final_features[:,hour_step]):
            all_none_steps += 1
            none_steps_flag = True
        else:
            break
            
    # update input window start: ICU first time --> first data point
    if none_steps_flag:
        first_hours += all_none_steps
        # exclude: first data point after ICU stay
        if first_hours >= last_hours:
            return features, False
        # update feature array & outcome time delta (onset - first time)
        final_features = final_features[:, all_none_steps:]
        if features['death_yn'] == 1:
            death_time = features['death_time'] - all_none_steps
            if death_time <= 0:
                return features, False
            features['death_time'] = death_time

        if features['cpr_yn'] == 1:
            cpr_time = [(i-all_none_steps, q-all_none_steps) for i, q in features['cpr_time']]
            cpr_time = [(i,q) for i,q in cpr_time if i >= 0 and q >= 0]
            if len(cpr_time) == 0:
                features['cpr_yn']   = 0
                features['cpr_time'] = None
            else:
                features['cpr_time'] = cpr_time

        if features['vasso_yn'] == 1:
            vasso_time = [(i-all_none_steps, q-all_none_steps) for i, q in features['vasso_time']]
            vasso_time = [(i,q) for i,q in vasso_time if i >= 0 and q >= 0]
            if len(vasso_time) == 0:
                features['vasso_yn']   = 0
                features['vasso_time'] = None
            else:
                features['vasso_time'] = vasso_time

        if features['intubation_yn'] == 1:
            intubation_time = [(i-all_none_steps, q-all_none_steps) for i, q in features['intubation_time']]
            intubation_time = [(i,q) for i,q in intubation_time if i >= 0 and q >= 0]
            if len(intubation_time) == 0:
                features['intubation_yn']   = 0
                features['intubation_time'] = None
            else:
                features['intubation_time'] = intubation_time

    # update features dict -- delete each {feature_name: list(value, time)}
    for i in feature_types:
        if i in features:
            del features[i]
    if 'GCS_EYE' in features:
        del features['GCS_EYE']
    if 'GCS_VER' in features:
        del features['GCS_VER']
    if 'GCS_MOT' in features:
        del features['GCS_MOT']
    # update: merge into {inputs: list(value)}
    features['inputs']              = final_features
    # update: existing features, input window range (first, last)
    features['feature_order']       = features_in_order
    features['window_first_idx_hr'] = first_hours
    features['window_last_idx_hr']  = last_hours

    return features, True

            
def write_feature(pat_id, chid, features):
    out_file = os.path.join(ARG.out_dir + "/{}".format(pat_id), '{}_{}.pkl'.format(pat_id, chid))
    with open(out_file, 'wb') as f:
        pickle.dump(features, f)


def extract_feature_each(pat_id, counts=None):
    path = os.path.join(ARG.in_dir, pat_id)
    
    # 환자 폴더 안에 폴더들은 내원번호들
    chid_list = [ chid for chid in os.listdir(path) if os.path.isdir(os.path.join(path, chid)) ] 

    # 각 label count 할때 사용
    # label_count = {'cpr':0, 'vasso':0, 'intubation':0, 'death':0, 'txt':0}

    for chid in chid_list:
        features                    = {'pat_id': pat_id, 'chid': chid}
        features, pat_flag          = read_patient_file(features, path, chid)
        features, obs_flag, counts  = read_observation_file(features, path, chid, counts)
        features, counts            = read_labresult_file(features, path, chid, counts)
        features, _                 = update_image_info(features, chid)
        features, _                 = update_txt_info(features, chid)
        features                    = extract_targets(features, path, chid)
        features, hourly_flag       = hourly_extract(features, path)
        # exclude: at least one flag is False
        if not (pat_flag and obs_flag and hourly_flag): 
            continue
        # print("features: ", features)

        # save features to .pkl file
        if not os.path.isdir(ARG.out_dir + "/{}".format(pat_id)):
            os.makedirs(ARG.out_dir + "/{}".format(pat_id))
        write_feature(pat_id, chid, features)
        # return counts

def main():
    pat_ids      = get_patids()
    ARG.img_dict = load_image_info()
    ARG.txt_dict = load_txt_info()
    
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
    #     counts = extract_feature_each(pat_id, counts)
    
    # print("final counts: ", counts)

if __name__ == '__main__':
    main()
