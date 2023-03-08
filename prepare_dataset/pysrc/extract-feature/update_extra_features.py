import os
import argparse
import pickle
import numpy as np
import pandas as pd

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/mnt/aitrics_ext/ext01/MedicalAI/data/extract-feature/SVRC/extract_feature')
parser.add_argument('--death_file', type=str, default='/mnt/aitrics_ext/ext01/MedicalAI/rawdata/SVRC/DEATH_20180824.csv')
parser.add_argument('--image_file', type=str, default='/mnt/aitrics_ext/ext01/MedicalAI/rawdata/SVRC/DICOM_Info.csv')
ARG = parser.parse_args()


def update_death_info():
    # update data files - death info
    df = pd.read_csv(ARG.death_file)
    df = df.sort_values(['UnitNo', 'ChosNo', 'ActDt'])
    df = df[~df['ChosNo'].duplicated(keep='last')] # remove duplicated cases, keep last time

    for _, row in df.iterrows():
        row = row.to_dict()
        uid, chid = row['UnitNo'], row['ChosNo']
        death_time = datetime.strptime(row['ActDt'].replace('.000',''), '%Y-%m-%d %H:%M:%S')
        
        datapath = os.path.join(ARG.data_dir, '{}_{}.pkl'.format(uid,chid))
        data = pickle.load(open(datapath,'rb'))
        data['Death_YN'] = 1
        data['Death_time'] = death_time
        pickle.dump(data, open(datapath,'wb'))


def update_image_info():
    # update data files - CXR info
    df = pd.read_csv(ARG.image_file)
    df = df[(df['OrdNm'] == 'Chest AP') | (df['OrdNm'] == 'Chest PA')]
    df = df.sort_values(['UnitNo', 'ChosNo', 'ExecYmd'])

    for _, row in df.iterrows():
        row = row.to_dict()
        uid, chid = row['UnitNo'], row['ChosNo']
        execdt = datetime.strptime('{}'.format(row['ExecYmd']), '%Y%m%d')
        imgpath = row['ImgPath'].replace('/ext01','')

        datapath = os.path.join(ARG.data_dir, '{}_{}.pkl'.format(uid,chid))
        data = pickle.load(open(datapath,'rb'))
        data['CXR'] = data.get('CXR', list())
        data['CXR'].append((imgpath, execdt))
        pickle.dump(data, open(datapath,'wb'))


def update_last_feature_time(data):
    feature_names = [
        'DBP', 'SBP', 'HR', 'RR', 'BT', 
        'Sat', 'GCS', 'WBC', 'PLT', 'Bilirubin', 
        'Creatinine', 'Lactate', 'Sodium', 'Potassium', 'Hematocrit', 
        'HCO3', 'CRP', 'pH', 'CXR',
    ] # 19 features + (Gender, Age)

    
    last_time_list = list()
    for feature in feature_names:
        feature_list = data.get(feature, list())
        if len(feature_list) == 0:
            continue
        feature_list = sorted(feature_list, key=lambda x: x[1])
        last_time_list.append(feature_list[-1][1])
    last_time_list.sort()

    if len(last_time_list) == 0:
        data['last_feature_time'] = ''
    else:
        data['last_feature_time'] = last_time_list[-1]
    data['Death_YN'] = data.get('Death_YN', 0)
    data['Death_time'] = data.get('Death_time', '')
    return data


def update_data_each(datapath):
    data = pickle.load(open(datapath,'rb'))
    data = update_last_feature_time(data)
    pickle.dump(data, open(datapath, 'wb'))


if __name__ == '__main__':
    
    update_death_info()
    update_image_info()
    
    from multiprocessing import Pool
    from tqdm import tqdm

    data_list = os.listdir(ARG.data_dir)
    data_list = [os.path.join(ARG.data_dir,fname) for fname in data_list]
    n_processes = min(os.cpu_count(), len(data_list))

    results = list()
    pool = Pool(processes=n_processes)
    for r in tqdm(pool.imap_unordered(update_data_each, data_list),
                    total=len(data_list), ncols=75):
        results.append(r)
    pool.close()
    pool.join()


