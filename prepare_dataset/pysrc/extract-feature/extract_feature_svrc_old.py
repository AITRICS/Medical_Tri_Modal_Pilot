import os, re
import argparse
import pickle
from struct import pack
import numpy as np
import csv
import re
import math
from datetime import datetime

from common.util.datetime_util import *
from common.util.process_util import *
from common.feature_table.feature_table_svrc import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir'         , type=str)
parser.add_argument('--in_dir'          , type=str)
parser.add_argument('--txt-path'          , type=str, default="/mnt/aitrics_ext/ext01/shared/multi_modal/rawdata/Clean_입원기록.csv")
ARG = parser.parse_args()
# CPR: CPR_20180824.csv, "MinTime"
# Vasso:  SEPSIS_DRUG_New_SurrogateKey_utf8.csv, "SugaCd" ("min_DrgExecStrYmdHms" - "max_DrgExecStrYmdHms")
# Intubation:  SEPSIS_Frm_CRRT_respiration_SurrogateKey_utf8.csv, Code: "A0006353", 시점: "AttrValue"

VASSO_CODE = ["D1NORP01", "D1NORP02", "D1DOPA01", "D1DOPA03", "D1DOBU01", "D1DOBU02", "D1VASO01"]
INTUBATION_IN_CODE = "A0006353"
INTUBATION_OUT_CODE = "A0006354"
CHEST_XRAY_INCLUSION = ['Chest AP', 'Chest PA']
GLOBAL_TEXT = {}

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

def read_patient_file(features, uid, chid):
    in_file = os.path.join(ARG.in_dir, uid, chid, 'PERSON_SurrogateKey.pkl')
    if not os.path.exists(in_file):
        return features, False

    with open(in_file, 'rb') as f:
        d = pickle.load(f)[0]

    features['gender'] = d['GENDER_CONCEPT_ID'].strip()
    features['age'] = float(d['AGE'].strip())
    features['admission_time'] = datetime.strptime(d['AdmiYmd'].strip(), '%Y%m%d')
    features['discharge_time'] = datetime.strptime(d['OtrmYmd'].strip(), '%Y%m%d')
    # print("features['gender']: ", features['gender'])
    # print("features['age']: ", features['age'])
    # print("features['admission_time']: ", features['admission_time'])
    # print("features['discharge_time']: ", features['discharge_time'])

    
    return features

def read_observation_file(features, uid, chid):
    in_file_bases = [
        'OBSERVATION_VS_SurrogateKey.pkl',
        'OBSERVATION_VS2014_SurrogateKey.pkl',
        'OBSERVATION_VS2015_SurrogateKey.pkl',
        'OBSERVATION_VS2016_SurrogateKey.pkl',
        'OBSERVATION_VS2017_SurrogateKey.pkl',
    ]

    extract_concept_ids = {
        '1000500001': 'HR',
        '1000600001': 'RR',
        '1000800001': 'SBP',
        '1000900001': 'DBP',
        '1000700001': 'BT',
        '2001100049': 'Sat',
        '5002500001': 'GCS',
    }

    for in_file_base in in_file_bases:
        in_file = os.path.join(ARG.in_dir, uid, chid, in_file_base)
        if not os.path.exists(in_file):
            continue

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['OBSERVATION_CONCEPT_ID']
            if context_id not in extract_concept_ids.keys():
                continue

            feature_dt = feature_item['ActDt']
            try:
                feature_dt = datetime.strptime(feature_dt,'%Y-%m-%d %H:%M')
            except:
                continue
                
            feature_code = extract_concept_ids[context_id]
            feature_value = feature_item['MODIFIED_VALUE_AS_NUMBER']
            
            if feature_code == 'GCS':
                v_norm = feature_value.lower()
                if v_norm == 'alert':
                    feature_value = 15
                elif v_norm == 'drowsy':
                    feature_value = 13
                elif v_norm == 'stupor':
                    feature_value = 10
                elif v_norm == 'semicoma':
                    feature_value = 6
                elif v_norm == 'coma':
                    feature_value = 3
                else:
                    continue
            else:
                try:
                    feature_value = float(feature_value)
                except Exception:
                    continue

            features[feature_code] = features.get(feature_code, list())
            features[feature_code].append((feature_value, feature_dt))
    
    return features

def read_labresult_file(features, uid, chid):
    extract_context_ids = {
        'T. Bilirubin[Serum]': 'Bilirubin',
        'Lactate[Arterial Whole blood]': 'Lactate',
        'Lactate[Venous Whole blood]': 'Lactate',
        'Na[Serum]': 'Sodium',
        'K[Serum]': 'Potassium',
        'Creatinine[Serum]': 'Creatinine',
        'Hct[Arterial Whole blood]': 'Hematocrit',
        'Hct[Whole blood]': 'Hematocrit',
        'WBC COUNT[Whole blood]': 'WBC',
        'HCO3-[Arterial Whole blood]': 'HCO3',
        'CRP (C-Reactive Protein)[Serum]': 'CRP',
        'pH[Arterial Whole blood]': 'pH'
    }
    
    in_file = os.path.join(ARG.in_dir, uid, chid, 'Lab2016_SurrogateKey.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)

    for feature_item in d:
        context_id = feature_item['preLabOrdNm']
        if context_id not in extract_context_ids.keys():
            continue

        feature_dt = feature_item['preLabOrdExecYmdHm']
        try:
            feature_dt = datetime.strptime(feature_dt,'%Y-%m-%d %H:%M')
        except:
            continue
            
        feature_code = extract_context_ids[context_id]
        try:
            feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['preLabNmrcRslt'])[0])
        except Exception:
            continue

        features[feature_code] = features.get(feature_code, list())
        features[feature_code].append((feature_value, feature_dt))
    
    in_file = os.path.join(ARG.in_dir, uid, chid, 'SEPSIS_LAB_New_SurrogateKey_utf8.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)
    
    for feature_item in d:
        context_id = feature_item['ExamNm']
        if context_id != 'PLT Count':
            continue
        
        try:
            feature_ymd = feature_item['OrdYmd']
            feature_hms = '{:06d}'.format(int(feature_item['OrdHms']))
            feature_dt = datetime.strptime(feature_ymd + feature_hms, '%Y%m%d%H%M%S')
        except:
            continue
        
        feature_code = 'PLT' 
        try:
            feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['NmrcRslt'])[0])
        except Exception:
            continue

        features[feature_code] = features.get(feature_code, list())
        features[feature_code].append((feature_value, feature_dt))

    return features

def write_features(features, pat_id, chid):
    out_file = os.path.join(ARG.out_dir, '{}_{}.pkl'.format(pat_id,chid))
    with open(out_file, 'wb') as f:
        pickle.dump(features, f)

def hourly_extract(features, path, chid):
    feature_types = ['HR', 'RR', 'BT', 'SBP', 'DBP', 'Sat',
                        'GCS', 'Hematocrit', 'PLT', 'WBC', 'Bilirubin', 'pH', 'HCO3', 
                        'Creatinine', 'Lactate', 'Potassium', 'Sodium', 'CRP']

    # 입원 시간 기준 첫시간과 마지막 시간
    first_time = datetime_to_hours(features['admission_time'])
    last_time = datetime_to_hours(str2datetime(features['discharge_time']))

    last_hours = int(math.ceil(last_time))
    first_hours = int(math.ceil(first_time))

    # 입원 시간 기준 첫시간기준으로 death_time 구함
    if features['death_yn'] == 1:
        death_time = max(features['death_time'])
        if death_time <= first_time:
            return features, False

        death_after = death_time - first_time
        features['death_time'] = death_after

        # death time 후에 input window 구간이 있을경우 겹칠경우 window size를 death time시간 바로전 까지로 수정
        if death_time <= last_hours:
            last_hours = int(math.ceil(death_time)) - 1

    # 올림 된 입원 시간 기준 첫시간기준으로 cpr, vasso, intubation 시간구간 구함
    if features['cpr_yn'] == 1:
        features['cpr_time'] = sorted(features['cpr_time'])
        cpr_after = [(np.round(i[0] - first_hours, 4), np.round(i[1] - first_hours, 4)) for i in features['cpr_time']]
        cpr_after = [(i, q) for i, q in cpr_after if i >= 0 and q >= 0]
        features['cpr_time'] = list(cpr_after)

    if features['vasso_yn'] == 1:
        features['vasso_time'] = sorted(features['vasso_time'])
        vasso_after = [(np.round(i[0] - first_hours, 4), np.round(i[1] - first_hours, 4)) for i in features['vasso_time']]
        intubation_after = [(i, q) for i, q in vasso_after if i >= 0 and q >= 0]
        features['vasso_time'] = list(vasso_after)

    if features['intubation_yn'] == 1:
        features['intubation_time'] = sorted(features['intubation_time'])
        intubation_after = [np.round(i - first_hours, 4) for i in features['intubation_time']]
        intubation_after = [i for i in intubation_after if i >= 0]
        features['intubation_time'] = list(intubation_after)

    all_features = []
    features_in_order = []
    # ceil 된 입원 시간 기준 첫 시간 ~ 마지막 시간 기준으로 
    for i in feature_types:
        if i in features:
            features_in_order.append(i)
            values_list = [q[0] for q in features[i]]
            time_list = [datetime_to_hours(q[1]) for q in features[i]]
            rd_time_list = [math.ceil(datetime_to_hours(q[1])) for q in features[i]]

            one_feature = []
            for hr_idx in range(first_hours, last_hours+1):
                d_feature = []
                while hr_idx in rd_time_list:
                    index = rd_time_list.index(hr_idx)
                    if len(d_feature) == 0:
                        d_feature = [values_list[index], time_list[index]]
                    else:
                        if abs(d_feature[1] - hr_idx) <= abs(time_list[index] - hr_idx):
                            pass
                        else:
                            d_feature = [values_list[index], time_list[index]]
                    del values_list[index]
                    del time_list[index]
                    del rd_time_list[index]
                
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
    #     for start in features['intubation_time']:
    #         start_hr = int(round(start))
    #         if start_hr >= len(intubation_samples):
    #             continue
    #         intubation_samples[start_hr] = 1

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
            features['intubation_time'] = [i-all_none_steps for i in features['intubation_time']]
            # intubation_samples = list(intubation_samples)[all_none_steps:]
    # else:
    #     cpr_samples = list(cpr_samples)
    #     vasso_samples = list(vasso_samples)
    #     intubation_samples = list(intubation_samples)

    for i in feature_types:
        if i in features:
            del features[i]
    if 'GCS_EYE' in features:
        del features['GCS_EYE']
    if 'GCS_VER' in features:
        del features['GCS_VER']
    if 'GCS_MOT' in features:
        del features['GCS_MOT']
    
    # features['cpr_inputs'] = cpr_samples
    # features['vasso_inputs'] = vasso_samples
    # features['intubation_inputs'] = intubation_samples
    features['inputs'] = final_features
    features['feature_order'] = features_in_order

    features['window_first_idx_hr'] = first_hours
    features['window_last_idx_hr'] = last_hours

    return features, True


def update_image_info(features, path, chid):
    image_events_file = os.path.join(path, chid, 'DICOM_Info.pkl')
    if not os.path.exists(image_events_file):
        features['cxr_yn'] = 0
        features['cxr_input'] = None
        return features, 0

    with open(image_events_file, 'rb') as f:
        cxr_infos = pickle.load(f)
    # print("cxr_infos: ", cxr_infos)
    ordnm_infos = [i for i in cxr_infos if i['OrdNm'] in CHEST_XRAY_INCLUSION]
    cxr_paths = [i['ImgPath'].replace('/ext01','') for i in ordnm_infos]
    cxr_times = [i['ExecYmd'] for i in ordnm_infos]
    
    cxr_final_infos = [[t, p] for t, p in sorted(zip(cxr_times, cxr_paths))]
    features['cxr_yn'] = 1
    features['cxr_input'] = cxr_final_infos

    return features, 1

def read_txt():
    txt_path = ARG.txt_path
    pattern = re.compile(r'(,\s){2,}')

    with open(txt_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if "주호소" not in row[4]:
                pass
            else:
                unit_chid = row[0] + "_" + row[1]
                if unit_chid not in GLOBAL_TEXT:
                    GLOBAL_TEXT[unit_chid] = [row[2], re.sub(' +', ' ', row[5])]

def update_txt_info(features, pat_id, chid):
    unit_chid = pat_id + "_" + chid
    if unit_chid in GLOBAL_TEXT:
        features['txt_input'] = GLOBAL_TEXT[unit_chid]
        # print(GLOBAL_TEXT[unit_chid])
        return features, 1
    else:
        features['txt_input'] = None
        return features, 0



def extract_targets(features, pat_id, chid):
    cpr_events_file = os.path.join(pat_id, chid, 'CPR_20180824.pkl')
    vasso_events_file = os.path.join(pat_id, chid, 'SEPSIS_DRUG_New_SurrogateKey_utf8.pkl')
    intubation_events_file = os.path.join(pat_id, chid, 'SEPSIS_Frm_CRRT_respiration_SurrogateKey_utf8.pkl')
    death_events_file = os.path.join(pat_id, chid, 'DEATH_20180824.pkl')

    flag = True

    label_count = {'cpr':0, 'vasso':0, 'intubation':0, 'death':0}
    if not os.path.exists(cpr_events_file):
        cpr_yn = False
    else:
        cpr_yn = True
        with open(cpr_events_file, 'rb') as f:
            cpr_infos = pickle.load(f)

    if not os.path.exists(vasso_events_file):
        vasso_yn = False
    else:
        vasso_yn = False
        with open(vasso_events_file, 'rb') as f:
            vasso_infos = pickle.load(f)
            vasso_infos = [i for i in vasso_infos if i['SugaCd'] in VASSO_CODE]
            if len(vasso_infos) > 0:
                vasso_yn = True
    
    if not os.path.exists(intubation_events_file):
        intubation_yn = False
    else:
        intubation_yn = False
        with open(intubation_events_file, 'rb') as f:
            intubation_infos = pickle.load(f)
            intubation_infos = [i for i in intubation_infos if i['AttrCd'] in INTUBATION_IN_CODE]
            # intubation_out_infos = [i for i in intubation_infos if i['AttrCd'] in INTUBATION_OUT_CODE]
            if len(intubation_infos) > 0:
                intubation_yn = True
                # print("1 intubation_in_infos: ", intubation_in_infos)
                # print("2 intubation_out_infos: ", intubation_out_infos)

    if not os.path.exists(death_events_file):
        death_yn = False
    else:
        death_yn = True
        with open(death_events_file, 'rb') as f:
            death_infos = pickle.load(f)

    if cpr_yn:
        cpr_flag = False
        for cpr_info in cpr_infos:
            cpr_min = str2datetime(cpr_info['MinTime'])
            cpr_max = str2datetime(cpr_info['MaxTime'])

            if cpr_min != NOT_CONVERTED and cpr_max != NOT_CONVERTED:
                cpr_begin_time = datetime_to_hours(cpr_min)
                cpr_end_time = datetime_to_hours(cpr_max)
                features['cpr_yn'] = 1
                features['cpr_time'] = features.get('cpr_time', list())
                features['cpr_time'].append((cpr_begin_time, cpr_end_time))
                cpr_flag = True

        if cpr_flag:
            label_count['cpr'] += 1
        else:
            features['cpr_yn'] = 0
            features['cpr_time'] = None
            flag = False
    else:
        features['cpr_yn'] = 0
        features['cpr_time'] = None

    if vasso_yn:
        vasso_flag = False
        for vasso_info in vasso_infos:
            vass_min = str2datetime(vasso_info['min_DrgExecStrYmdHms'])
            vass_max = str2datetime(vasso_info['max_DrgExecStrYmdHms'])

            if vass_min != NOT_CONVERTED and vass_max != NOT_CONVERTED:
                vasso_begin_time = datetime_to_hours(vass_min)
                vasso_end_time = datetime_to_hours(vass_max)
                features['vasso_yn'] = 1
                features['vasso_time'] = features.get('vasso_time', list())
                features['vasso_time'].append((vasso_begin_time, vasso_end_time))
                vasso_flag = True

        if vasso_flag:
            label_count['vasso'] += 1
        else:
            features['vasso_yn'] = 0
            features['vasso_time'] = None
            flag = False

    else:
        features['vasso_yn'] = 0
        features['vasso_time'] = None
    
    if intubation_yn:
        intubation_flag = False
        for intubation_info in intubation_infos:
            intubation_min = str2datetime(intubation_info['AttrValue'])

            if intubation_min != NOT_CONVERTED:
                intubation_begin_time = datetime_to_hours(intubation_min)
                features['intubation_yn'] = 1
                features['intubation_time'] = features.get('intubation_time', list())
                features['intubation_time'].append(intubation_begin_time)
                intubation_flag = True
        
        if intubation_flag:
            label_count['intubation'] += 1
        else:
            features['intubation_yn'] = 0
            features['intubation_time'] = None
            flag = False

    else:
        features['intubation_yn'] = 0
        features['intubation_time'] = None

    if death_yn:
        death_flag = False
        for death_info in death_infos:
            death_min = str2datetime(death_info['ActDt'])

            if death_min != NOT_CONVERTED:
                death_time = datetime_to_hours(death_min)
                features['death_yn'] = 1
                features['death_time'] = features.get('death_time', list())     # in case it was recorded multiple times by mistake
                features['death_time'].append(death_time)
                death_flag = True

        if death_flag:
            label_count['death'] += 1
        else:
            features['death_yn'] = 0
            features['death_time'] = None
            flag = False

    else:
        features['death_yn'] = 0
        features['death_time'] = None

    return features, label_count, flag
    
    

def extract_features_each(pat_id):
    path = ARG.in_dir + "/" + str(pat_id)
    chid_id_list = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ] # 1 or more files can exist
    
    label_count = {'cpr':0, 'vasso':0, 'intubation':0, 'death':0, 'img':0, 'txt':0}

    for chid in chid_id_list:
        features = {'pat_id': pat_id, 'chid': chid}
        features = read_patient_file(features, pat_id, chid)

        features = read_observation_file(features, pat_id, chid)

        features = read_labresult_file(features, pat_id, chid)

        features, img_count = update_image_info(features, path, chid)

        features, txt_count = update_txt_info(features, pat_id, chid)

        features, l_count, flag = extract_targets(features, path, chid)
        if flag == False:
            continue

        label_count['cpr'] += l_count['cpr']
        label_count['vasso'] += l_count['vasso']
        label_count['intubation'] += l_count['intubation']
        label_count['death'] += l_count['death']
        label_count['img'] += img_count
        label_count['txt'] += txt_count

        features, flag = hourly_extract(features, path, chid)
        
        write_features(features, pat_id, chid)

    return label_count

def main():
    pat_ids = get_patids()
    print("1. Number of patients: ", len(pat_ids))
    label_count = {'cpr':0, 'vasso':0, 'intubation':0, 'death':0, 'img':0, 'txt':0}
    read_txt()

    run_multi_process(extract_features_each, pat_ids)

    # counts = None
    # for idx, pat_id in enumerate(pat_ids):
    #     print("{} / {}".format(str(idx+1), str(len(pat_ids))))
    #     l_count = extract_features_each(pat_id)
    #     if l_count['cpr'] > 0:
    #         label_count['cpr'] += 1
    #     if l_count['vasso'] > 0:
    #         label_count['vasso'] += 1
    #     if l_count['intubation'] > 0:
    #         label_count['intubation'] += 1
    #     if l_count['death'] > 0:
    #         label_count['death'] += 1
    #     if l_count['img'] > 0:
    #         label_count['img'] += 1
    #     if l_count['txt'] > 0:
    #         label_count['txt'] += 1

    # print("label_count: ", label_count)
    
if __name__ == '__main__':
    main()

# if __name__ == '__main__':
    
#     from multiprocessing import Pool
#     from tqdm import tqdm

#     uid_list = os.listdir(ARG.in_dir)
#     n_processes = min(os.cpu_count(), len(uid_list))
    
#     results = list()
#     pool = Pool(processes=n_processes)
#     for r in tqdm(pool.imap_unordered(extract_features_by_uid, uid_list),
#                   total=len(uid_list), ncols=75):
#         results.append(r)
#     pool.close()
#     pool.join()