import os
import pickle
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument('--out_dir', type=str, default='/mnt/aitrics_ext/ext01/alex/Medical_Multi_Modal_V1/prepare_dataset/data')
# parser.add_argument('--in_dir', type=str, default='/mnt/aitrics_ext/ext01/alex/Medical_Multi_Modal_V1/prepare_dataset/data/extract-features')
parser.add_argument('--out_dir', type=str, default='/home/claire/Medical_Multi_Modal_V1/prepare_dataset/script/extract-feature')
parser.add_argument('--in_dir', type=str, default='/mnt/aitrics_ext/ext01/MedicalAI/data/extract-feature/MIMIC/extract_feature')
ARG = parser.parse_args()


def extract_data_stats_svrc(datapath):
    data = pickle.load(open(datapath,'rb'))
    filename = datapath.split('/')[-1]
    out_row = dict(
        UnitNo = filename.split('_')[0],
        ChosNo = filename.split('_')[1].replace('.pkl',''),
        Death_YN = data['Death_YN'],
        Gender = data['Gender'],
        Age = data['Age'],
        Admission_time = data['Admission_time'],
        Discharge_time = data['Discharge_time'],
        Death_time = data['Death_time'],
    )
    if 'last_feature_time' in data.keys():
        out_row['last_feature_time'] = data['last_feature_time']

    # check CXR data
    image_list = data.get('CXR', list())
    out_row['CXR_freq'] = len(image_list)

    feature_names = [
        'DBP', 'SBP', 'HR', 'RR', 'BT', 
        'Sat', 'GCS', 'WBC', 'PLT', 'Bilirubin', 
        'Creatinine', 'Lactate', 'Sodium', 'Potassium', 'Hematocrit', 
        'HCO3', 'CRP', 'pH',
    ] # 19 features + (Gender, Age)
    for feature in feature_names:
        feature_list = data.get(feature, list())
        if len(feature_list) == 0:
            out_row['{}_freq'.format(feature)] = 0
            out_row['{}_mean'.format(feature)] = np.nan
            out_row['{}_std'.format(feature)] = np.nan
            out_row['{}_min'.format(feature)] = np.nan
            out_row['{}_max'.format(feature)] = np.nan
            out_row['{}_q1'.format(feature)] = np.nan
            out_row['{}_q2'.format(feature)] = np.nan
            out_row['{}_q3'.format(feature)] = np.nan
            continue
        
        fval_list = [fval[0] for fval in feature_list]
        fval_percentile = np.percentile(fval_list, [0, 25, 50, 75, 100], interpolation='nearest')
        out_row['{}_freq'.format(feature)] = len(feature_list)
        out_row['{}_mean'.format(feature)] = np.mean(fval_list)
        out_row['{}_std'.format(feature)] = np.std(fval_list)
        out_row['{}_min'.format(feature)] = fval_percentile[0]
        out_row['{}_max'.format(feature)] = fval_percentile[4]
        out_row['{}_q1'.format(feature)] = fval_percentile[1]
        out_row['{}_q2'.format(feature)] = fval_percentile[2]
        out_row['{}_q3'.format(feature)] = fval_percentile[3]

    return out_row

def extract_data_stats_mimic4(datapath):
    data = pickle.load(open(datapath,'rb'))

    filename = datapath.split('/')[-1]
    out_row = dict(
        UnitNo = data['UnitNo'],
        ChosNo = data['CHID'],
        Death_YN = data['Death_YN'],
        Gender = data['Gender'],
        Age = data['Age'],
        Admission_time = data['Admission_time'],
        Discharge_time = data['Discharge_time'],
        Death_time = data['Death_time'],
    )
    # if 'last_feature_time' in data.keys():
    #     out_row['last_feature_time'] = data['last_feature_time']

    # check CXR data
    image_list = data.get('CXR', list())
    out_row['CXR_freq'] = len(image_list)

    feature_names = [
        'DBP', 'SBP', 'PULSE', 'RESP', 'TEMP', 
        'SpO2', 'GCS', 'WBC', 'PLATELET', 'BILIRUBIN', 
        'CREATININE', 'LACTATE', 'SODIUM', 'POTASSIUM', 'HEMATOCRIT', 
        'HCO3', 'CRP', 'pH',
    ] # 19 features + (Gender, Age)
    # print("data: ", data)

    for feature in feature_names:
        feature_list = data.get(feature, list())
        if len(feature_list) == 0:
            out_row['{}_freq'.format(feature)] = 0
            out_row['{}_mean'.format(feature)] = np.nan
            out_row['{}_std'.format(feature)] = np.nan
            out_row['{}_min'.format(feature)] = np.nan
            out_row['{}_max'.format(feature)] = np.nan
            out_row['{}_q1'.format(feature)] = np.nan
            out_row['{}_q2'.format(feature)] = np.nan
            out_row['{}_q3'.format(feature)] = np.nan
            continue

        fval_list = [fval[0] for fval in feature_list]
        fval_percentile = np.percentile(fval_list, [0, 25, 50, 75, 100], interpolation='nearest')
        out_row['{}_freq'.format(feature)] = len(feature_list)
        out_row['{}_mean'.format(feature)] = np.mean(fval_list)
        out_row['{}_std'.format(feature)] = np.std(fval_list)
        out_row['{}_min'.format(feature)] = fval_percentile[0]
        out_row['{}_max'.format(feature)] = fval_percentile[4]
        out_row['{}_q1'.format(feature)] = fval_percentile[1]
        out_row['{}_q2'.format(feature)] = fval_percentile[2]
        out_row['{}_q3'.format(feature)] = fval_percentile[3]

    return out_row


if __name__ == '__main__':
    
    from multiprocessing import Pool
    from tqdm import tqdm

    data_list = os.listdir(ARG.in_dir)
    data_list = [os.path.join(ARG.in_dir,fname) for fname in data_list]

    results = list()
    
    n_processes = min(os.cpu_count(), len(data_list))
    print(n_processes)
    results = list()
    pool = Pool(processes=n_processes)
    # for r in tqdm(pool.imap_unordered(extract_data_stats_svrc, data_list),
    for r in tqdm(pool.imap_unordered(extract_data_stats_mimic4, data_list),
                  total=len(data_list), ncols=40):
        results.append(r)
    pool.close()
    pool.join()

    # write stats
    df = pd.DataFrame.from_records(results)
    df.to_csv(os.path.join(ARG.out_dir, 'stats.csv'),
              encoding='utf-8-sig', index=False)
