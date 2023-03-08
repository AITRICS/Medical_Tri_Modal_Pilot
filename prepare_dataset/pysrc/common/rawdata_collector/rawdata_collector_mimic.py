# coding: utf-8

import os
import pickle
from datetime import timedelta

from common.util.datetime_util import *
from typing import *
from common.feature_table.feature_table_mimic import *

class Transfer:
    def __init__(self, d=None):
        if d:
            self.ward = d['ward']
            self.in_time = d['in_time']
            self.out_time = d['out_time']


class RawdataCollector:
    def __init__(self, arg, chid):
        self.arg = arg
        self.chid = chid

    def read_patient_file(self):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'ADMISSIONS.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        adm_time = d['ADMITTIME']
        discharge_time = d['DISCHTIME']

        adm_time = str2datetime(adm_time)
        discharge_time = str2datetime(discharge_time)
        discharge_time += timedelta(days=1)

        patinfo = {
            'chid': self.chid,
            'adm_time': adm_time,
            'discharge_time': discharge_time,
        }

        return True, patinfo

    def read_death_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.in_dir, self.chid, 'ADMISSIONS.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        if not int(d['HOSPITAL_EXPIRE_FLAG']):
            return False, None

        dod_time = str2datetime(d['DEATHTIME'])
        # dos_time = str2datetime(d['DOD_SSN'])

        if dod_time != NOT_CONVERTED:
            return True, dod_time
        # elif doh_time != NOT_CONVERTED:
        #     return True, doh_time
        # elif dos_time != NOT_CONVERTED:
        #     return True, dos_time
        else:
            return False, None

    def read_transfer_file(self) -> List[Transfer]:
        transfers = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'TRANSFERS.pkl')
        if not os.path.exists(in_file):
            return transfers

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            icu_in_time = str2datetime(row['INTIME'])
            icu_out_time = str2datetime(row['OUTTIME'])

            if icu_in_time == NOT_CONVERTED or icu_out_time == NOT_CONVERTED:
                continue

            transfer = Transfer(dict(
                ward='ICU',
                in_time=icu_in_time,
                out_time=icu_out_time,
            ))
            transfers.append(transfer)

        return transfers

    def read_sepsis_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.sepsis_dir, '{}.pkl'.format(self.chid))
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d: datetime = pickle.load(f)

        onset_time = str2datetime(d)

        return True, onset_time

    def read_aki_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.aki_dir, '{}.pkl'.format(self.chid))
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d: datetime = pickle.load(f)

        onset_time = str2datetime(d['chart_time'])

        return True, onset_time

    # def read_dtr_file(self) -> Tuple[bool, Any]:
    #     in_file = os.path.join(self.arg.dtr_dir, '{}.pkl'.format(self.chid))
    #     if not os.path.exists(in_file):
    #         return False, None

    #     with open(in_file, 'rb') as f:
    #         d: datetime = pickle.load(f)

    #     onset_time = str2datetime(d)

    #     return True, onset_time

    def read_cpr_file(self):
        #ITEMID: 225466
        #FILE: procedureevents_mv
        cpr_times = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'PROCEDUREEVENTS_MV.pkl')
        if not os.path.exists(in_file):
            return cpr_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            if row['ITEMID'] == '225466':
                cpr_time = str2datetime(row['STARTTIME'])
                cpr_times.append(cpr_time)

        return cpr_times

    def read_observation_file(self, feature_times):
        in_file_bases = [
        'CHARTEVENTS.pkl',
        ]

        for in_file_base in in_file_bases:
            in_file = os.path.join(self.arg.in_dir, self.chid, in_file_base)
            if not os.path.exists(in_file):
                continue

            with open(in_file, 'rb') as f:
                d = pickle.load(f)

            for feature_item in d:
                context_id = feature_item['ITEMID']
                if context_id not in FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION:
                    continue

                feature_code = FEATURE_FOR_EXTRACT_FEATURE.OBSERVATION[context_id]

                feature_time = str2datetime(feature_item['CHARTTIME'])
                if feature_time == NOT_CONVERTED:
                    continue

                feature_value = feature_item['VALUENUM']

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

                if feature_code == 'TEMP_C':
                    feature_code = 'TEMP'
                elif feature_code == 'TEMP_F':
                    feature_code = 'TEMP'
                    feature_value = (feature_value - 32.0) / 1.8
                
                feature_times.add(feature_time)

        return feature_times

    def read_labresult_file(self, feature_times):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'LABEVENTS.pkl')
        if not os.path.exists(in_file):
            return feature_times

        with open(in_file, 'rb') as f:
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

                feature_times.add(feature_time)

        return feature_times
        
