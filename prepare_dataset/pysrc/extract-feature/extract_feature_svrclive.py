# coding: utf-8

import os
import argparse
import pickle

from common.util.datetime_util import *
from common.util.process_util import *
from common.feature_table.feature_table_svrclive import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir'         , type=str)
parser.add_argument('--in_dir'          , type=str)
ARG = parser.parse_args()


def get_chids():
    return os.listdir(ARG.in_dir)


def read_patient_file(features, chid):
    in_file = os.path.join(ARG.in_dir, chid, 'emr_patient.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)[0]

        birth_dt = str2datetime(d['birth'])
        if birth_dt != NOT_CONVERTED:
            features['BIRTH'] = birth_dt

    return features


def read_observation_file(features, chid):
    in_file = os.path.join(ARG.in_dir, chid, 'emr_observation.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)

    for feature_item in d:
        context_id = feature_item['context_id']
        if context_id not in FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION:
            continue

        feature_code = FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION[context_id]
        feature_time = str2datetime(feature_item['observe_dt'])
        try:
            feature_value = float(feature_item['value'])
        except Exception:
            continue

        features[feature_code] = features.get(feature_code, list())
        features[feature_code].append((feature_time, feature_value))

    return features


def read_labresult_file(features, chid):
    in_file = os.path.join(ARG.in_dir, chid, 'emr_labresult.pkl')
    if not os.path.exists(in_file):
        return features

    with open(in_file, 'rb') as f:
        d = pickle.load(f)

    for feature_item in d:
        context_id = feature_item['code']
        if context_id not in FEATURE_FOR_EXTRACT_FEATURE.LAB:
            continue

        feature_code = FEATURE_FOR_EXTRACT_FEATURE.LAB[context_id]
        feature_dt = str2datetime(feature_item['result_dt'])

        try:
            feature_value = float(feature_item['value'])
        except Exception:
            continue

        features[feature_code] = features.get(feature_code, list())
        features[feature_code].append((feature_dt, feature_value))

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
    write_feature(chid, features)


def main():
    chids = get_chids()
    run_multi_process(extract_feature_each, chids)


if __name__ == '__main__':
    main()
