# coding: utf-8

import os
import argparse
import pickle
import re

from common.util.datetime_util import *
from common.util.process_util import *
from common.feature_table.feature_table_ilsan import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str)
parser.add_argument('--in_dir', type=str)
parser.add_argument('--patient_dir', type=str)
ARG = parser.parse_args()

# OBSERVATION_TO_FEATURE_MAP = dict([
#     ('TPR1100000110000041000021', 'PULSE'),
#     ('TPR1100000110000041000022', 'RESP'),
#     ('TPR1100000110000041000018', 'SBP'),
#     ('TPR1100000110000041000020', 'DBP'),
#     ('TPR1100000110000041000023', 'TEMP'),
#     ('TPR1100000010000031000004', 'SpO2'),
#     ('TPR1100000310000241000143', 'GCS'),
# ])
#
# LABRESULT_TO_FEATURE_MAP = dict([
#     ('Hct', 'HEMATOCRIT'),
#     ('Creatinine', 'CREATININE'),
#     ('Platelet count', 'PLATELET'),
# ])
#
# NO_DEATH = 'NO_DEATH'
# NO_PATIENT_FILE = 'NO_PATIENT_FILE'


def get_chids():
    return os.listdir(ARG.in_dir)


def read_patient_file(features, chid):
    filename = os.path.join(ARG.in_dir, chid, 'PERSON.pkl')
    if not os.path.exists(filename):
        return features

    with open(filename, 'rb') as f:
        d = pickle.load(f)[0]

    birth_y = int(d['YEAR_OF_BIRTH'])
    birth_m = int(d['MONTH_OF_BIRTH'])

    birth_dt = str2datetime('{}{}01'.format(birth_y, birth_m))

    if birth_dt != NOT_CONVERTED:
        features['BIRTH'] = birth_dt

    return features


def read_observation_file(features, chid):
    in_file_bases = [
        'OBSERVATION_201701.pkl',
        'OBSERVATION_201702.pkl',
        'OBSERVATION_201703.pkl',
        'OBSERVATION_201704.pkl',
        'OBSERVATION_201705.pkl',
        'OBSERVATION_201706.pkl',
        'OBSERVATION_201707.pkl',
        'OBSERVATION_201708.pkl',
        'OBSERVATION_201709.pkl',
        'OBSERVATION_201710.pkl',
        'OBSERVATION_201711.pkl',
        'OBSERVATION_201712.pkl',
    ]

    for in_file_base in in_file_bases:
        in_file = os.path.join(ARG.in_dir, chid, in_file_base)
        if not os.path.exists(in_file):
            continue

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['OBSERVATION_CONCEPT_ID']
            if context_id not in FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION:
                continue

            feature_code = FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION[context_id]
            feature_time = str2datetime(feature_item['OBSERVATION_DATETIME'])
            if feature_time == NOT_CONVERTED:
                continue

            feature_value = feature_item['OBSERVATION_SOURCE_VALUE']

            if feature_code == FEATURE_CODE.GCS:
                v_norm = feature_value.lower()
                if v_norm in {'a', 'alert'}:
                    feature_value = 15
                elif v_norm in {'d', 'drowsy', 'sed', 'd~dd', 'dd~d',
                                'confuse', 'conf', 'confu', 'confus', 'con',
                                'dd', 'dd~confuse'}:
                    feature_value = 13
                elif v_norm in {'stupor', 'st', 'dst', 'd.st', 'd st', 'dd-st'}:
                    feature_value = 10
                elif v_norm in {'semicoma', 'sc'}:
                    feature_value = 6
                elif v_norm in {'coma', 'c'}:
                    feature_value = 3
                else:
                    continue
            else:
                try:
                    feature_value = float(feature_value)
                except Exception:
                    continue

            features[feature_code] = features.get(feature_code, list())
            features[feature_code].append((feature_time, feature_value))

    return features


def read_labresult_file(features, chid):
    in_file_bases = [
        'LAB_201701.pkl',
        'LAB_201704.pkl',
        'LAB_201707.pkl',
        'LAB_201710.pkl',
    ]

    for in_file_base in in_file_bases:
        in_file = os.path.join(ARG.in_dir, chid, in_file_base)
        if not os.path.exists(in_file):
            continue

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['preLabOrdNm']
            if context_id not in FEATURE_FOR_EXTRACT_FEATURE.LAB:
                continue
            feature_code = FEATURE_FOR_EXTRACT_FEATURE.LAB[context_id]

            feature_time = str2datetime(feature_item['preLabOrdExecYmdHm'])
            if feature_time == NOT_CONVERTED:
                continue

            try:
                # feature_value = float(feature_item['preLabNmrcRslt'])
                feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['preLabNmrcRslt'])[0])
            except Exception:
                continue

            if feature_code == FEATURE_CODE.CRP:
                feature_value *= 10.0

            if feature_code == FEATURE_CODE.DDIMER:
                feature_value /= 1000.0

            features[feature_code] = features.get(feature_code, list())
            features[feature_code].append((feature_time, feature_value))

    return features


def write_feature(chid, features, transfers):
    outfile = os.path.join(ARG.out_dir, '{}.pkl'.format(chid))
    with open(outfile, 'wb') as f:
        pickle.dump(features, f)


def extract_feature_each(chid):
    features = {'CHID': chid}
    features = read_patient_file(features, chid)
    features = read_observation_file(features, chid)
    features = read_labresult_file(features, chid)
    transfers = read_transfer_file(chid)

    write_feature(chid, features, transfers)


def main():
    chids = get_chids()
    run_multi_process(extract_feature_each, chids)


if __name__ == '__main__':
    main()
