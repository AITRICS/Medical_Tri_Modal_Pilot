# -*- coding: utf-8 -*-
# @Author: steve
# @Date:   2022-03-03 16:37:54
# @Last Modified by:   steve
# @Last Modified time: 2022-03-03 19:32:10

import os
import csv
import argparse
import pickle as pkl

from tqdm import tqdm
from module_common import *
from common.util.hash_util import *
from common.util.process_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str)
parser.add_argument('--in_dir', type=str)
parser.add_argument('--rawdata', type=str)
parser.add_argument('--dataset', type=str)
ARG = parser.parse_args()


def get_unit_dic():
    patient_file = 'PERSON_SurrogateKey.csv' if ARG.dataset == 'SVRC' else 'TRANSFERS.csv'
    patient_file = os.path.join(ARG.rawdata, patient_file)

    unit_field = 'UnitNo' if ARG.dataset == 'SVRC' else 'SUBJECT_ID'
    chid_field = 'ChosNo' if ARG.dataset == 'SVRC' else 'HADM_ID'

    reader = csv.DictReader(open(patient_file, 'r', encoding='utf-8-sig'))

    chid_dict = dict()
    unit_set = set()

    for row in tqdm(reader, desc='make chid dict', ncols=75):
        chid = row[chid_field]
        unit = row[unit_field]
        if chid == '':
            continue
        chid_dict[chid] = unit
        unit_set.add(unit)

    return chid_dict, unit_set


def make_unit_dir(unit_set):
    for unit in tqdm(unit_set, desc='make unit dirs', ncols=75):
        out_unit_dir = os.path.join(ARG.out_dir, unit)
        os.makedirs(out_unit_dir)


def merge_by_unit(chid):
    if chid in CHID_DICT:
        unit = CHID_DICT[chid]
        # print('{} {}/'.format(os.path.join(ARG.in_dir, chid), os.path.join(ARG.out_dir, unit)))
        os.system("mv {} {}/".format(os.path.join(ARG.in_dir, chid), os.path.join(ARG.out_dir, unit)))

def split_by_subject_id(file_name):
    file_path = os.path.join(ARG.rawdata, file_name)
    reader = csv.DictReader(open(file_path, 'r', encoding='utf-8-sig'))
    subject_dict = dict()

    missing_subject_id = list()

    for row in reader:
        subject_id = row['SUBJECT_ID']
        subject_dict[subject_id] = subject_dict.get(subject_id, list())
        subject_dict[subject_id].append(row)

    for subject_id in list(subject_dict.keys()):
        out_dir = os.path.join(ARG.out_dir, subject_id)
        if not os.path.exists(out_dir):
            missing_subject_id.append(subject_id)
            continue
            
        out_path = os.path.join(out_dir, file_name.replace('.csv', '.pkl'))
        with open(out_path, 'wb') as fp:
            pkl.dump(subject_dict[subject_id], fp)

    return (file_name, missing_subject_id)

def mimic_subject_only_files():
    subject_only_files = [
        'POE_DETAIL.csv',
        'PATIENTS.csv',
        'EMAR_DETAIL.csv',
        'CXR_RECORD.csv',
        'CXR_STUDY.csv',
    ]

    missing_subject_id = run_multi_process(split_by_subject_id, subject_only_files)
    return missing_subject_id

def main():
    chid_dict, unit_set = get_unit_dic()
    global CHID_DICT
    CHID_DICT = chid_dict
    chids = list(chid_dict.keys())
    
    make_unit_dir(unit_set)
    run_multi_process(merge_by_unit, chids)

    if ARG.dataset == 'MIMIC':
        missing_subject_id = mimic_subject_only_files()

        if not os.path.exists('MIMIC_LOG'):
            os.mkdir('MIMIC_LOG')

        for (file_name, missing_ids) in missing_subject_id:
            writer = open('MIMIC_LOG/{}_missing_id.txt'.format(file_name.replace('.csv', '')),'w')
            writer.write('MISSING SUBJECT IDS\n')
            for subject_id in missing_ids:
                writer.write('{}\n'.format(subject_id))


if __name__ == '__main__':
    main()
