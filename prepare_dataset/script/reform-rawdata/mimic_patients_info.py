import os
import pickle
import csv
from tqdm import tqdm


if __name__ == '__main__':
    split_chid_path = '/mnt/aitrics_ext/ext01/steve/working/VitalCare-Model-v2/data/reform-rawdata/MIMIC/split_by_chid'
    chids = os.listdir(split_chid_path)
    patient_dir = os.path.dirname(split_chid_path)
    
    patients_dic = dict()
    reader = csv.DictReader(open('/mnt/aitrics_ext/ext01/steve/working/VitalCare-Model-v2/rawdata/MIMIC/PATIENTS.csv', 'r'))
    for row in reader:
        patients_dic[row['SUBJECT_ID']] = row

    for chid in tqdm(chids, total=len(chids), ncols=75):
        admissions = pickle.load(open(os.path.join(split_chid_path, chid, 'ADMISSIONS.pkl'), 'rb'))[0]
        subject_id = admissions['SUBJECT_ID']
        patient_info = patients_dic[subject_id]
        out_dir = os.path.join(split_chid_path, chid, 'PATIENTS.pkl')
        with open(out_dir, 'wb') as fp:
            pickle.dump(patient_info, fp)

