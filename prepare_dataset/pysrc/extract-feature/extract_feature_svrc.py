# coding: utf-8

import os
import argparse
import pickle
import re

from common.util.datetime_util import *
from common.util.process_util import *
from common.feature_table.feature_table_svrc import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str)
parser.add_argument('--in_dir', type=str)
ARG = parser.parse_args()


def get_chids():
    return os.listdir(ARG.in_dir)


def read_patient_file(features, chid):
    in_file = os.path.join(ARG.in_dir, chid, 'PERSON_SurrogateKey.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)
        d = d[0]

    birth_y = d['YEAR_OF_BIRTH'].strip()
    birth_m = d['MONTH_OF_BIRTH'].strip()
    birth_d = d['DAY_OF_BIRTH'].strip()

    birth_dt = str2datetime('{}{}{}'.format(birth_y, birth_m, birth_d))
    if birth_dt != NOT_CONVERTED:
        features['BIRTH'] = birth_dt

    return features

    # transfer_file = os.path.join(ARG.in_dir, chid, 'TIME_SurrogateKey.pkl')
    # if not os.path.exists(transfer_file):
    #     features['TRANSFER'] = 'NOT_CONVERTED'
    #     return features

    # with open(transfer_file, 'rb') as f:
    #     d = pickle.load(f)
    #     d = d[0]

    # icu_in = str2datetime(d['RgtDt'])
    # icu_out = str2datetime(d['LstUpdDt'])

    # if icu_in == 'NOT_CONVERTED' or icu_out == 'NOT_CONVERTED':
    #     features['TRANSFER'] = 'NOT_CONVERTED'
    #     return features
    # else:
    #     features['TRANSFER'] = {'ICU_IN': icu_in, 'ICU_OUT': icu_out}
    #     return features


def read_observation_file(features, chid):
    in_file_bases = [
        'OBSERVATION_VS_SurrogateKey.pkl',
        'OBSERVATION_VS2014_SurrogateKey.pkl',
        'OBSERVATION_VS2015_SurrogateKey.pkl',
        'OBSERVATION_VS2016_SurrogateKey.pkl',
        'OBSERVATION_VS2017_SurrogateKey.pkl',
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

            feature_dt = str2datetime(feature_item['ActDt'])
            if feature_dt == NOT_CONVERTED:
                continue

            feature_value = feature_item['MODIFIED_VALUE_AS_NUMBER']

            if feature_code == FEATURE_CODE.GCS:
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
            features[feature_code].append((feature_dt, feature_value))

    return features


def read_labresult_file(features, chid):
    in_file = os.path.join(ARG.in_dir, chid, 'Lab2016_SurrogateKey.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)

    for feature_item in d:
        context_id = feature_item['preLabOrdNm']
        if context_id not in FEATURE_FOR_EXTRACT_FEATURE.LAB:
            continue

        feature_code = FEATURE_FOR_EXTRACT_FEATURE.LAB[context_id]

        feature_dt = str2datetime(feature_item['preLabOrdExecYmdHm'])
        if feature_dt == NOT_CONVERTED:
            continue

        try:
            # feature_value = float(feature_item['preLabNmrcRslt'])
            feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['preLabNmrcRslt'])[0])
        except Exception:
            continue

        if feature_code == FEATURE_CODE.DDIMER:
            feature_value /= 1000.0

        features[feature_code] = features.get(feature_code, list())
        features[feature_code].append((feature_dt, feature_value))

    return features


def read_labresult_new_file(features, chid):
    in_file = os.path.join(ARG.in_dir, chid, 'SEPSIS_LAB_New_SurrogateKey_utf8.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['ExamNm']
            if context_id not in FEATURE_FOR_EXTRACT_FEATURE.LAB_NEW:
                continue

            feature_code = FEATURE_FOR_EXTRACT_FEATURE.LAB_NEW[context_id]
            feature_ymd = feature_item['OrdYmd']
            feature_hms = feature_item['OrdHms']

            feature_time = str2datetime('{}{}'.format(feature_ymd, feature_hms))
            if feature_time == NOT_CONVERTED:
                continue

            try:
                # feature_value = float(feature_item['NmrcRslt'])
                feature_value = float(re.findall("\d+\.\d+|\d+", feature_item['NmrcRslt'])[0])
            except Exception:
                continue

            if feature_code == FEATURE_CODE.DDIMER:
                feature_value /= 1000.0

            features[feature_code] = features.get(feature_code, list())
            features[feature_code].append((feature_time, feature_value))

    return features


def write_feature(chid, features):
    out_file = os.path.join(ARG.out_dir, '{}.pkl'.format(chid))
    with open(out_file, 'wb') as f:
        pickle.dump(features, f)


def extract_feature_each(chid):
    features = {'CHID': chid}
    features = read_patient_file(features, chid)
    features = read_observation_file(features, chid)
    features = read_labresult_file(features, chid)
    features = read_labresult_new_file(features, chid)
    write_feature(chid, features)


def main():
    chids = get_chids()
    features = run_multi_process(extract_feature_each, chids)


if __name__ == '__main__':
    features = main()
