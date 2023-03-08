# coding: utf-8

import os
import pickle
from datetime import timedelta

from common.util.datetime_util import *
from typing import *
from common.feature_table.feature_table_svrc import *


class Transfer:
    def __init__(self, d=None):
        if d:
            self.ward = d['ward']
            self.ward_type = d['ward_type']
            self.in_time = d['in_time']
            self.out_time = d['out_time']

class Operation:
    def __init__(self, d=None):
        if d:
            self.op_time = d['op_time']
            self.emergence = d['emergence']
            


class RawdataCollector:
    def __init__(self, arg, chid):
        self.arg = arg
        self.chid = chid

    def read_patient_file(self):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'SEPSIS_KN_PERSON_SurrogateKey.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        adm_time = d['AdmiDate']
        discharge_time = '99991231'

        adm_time = str2datetime(adm_time)
        discharge_time = str2datetime(discharge_time)

        patinfo = {
            'chid': self.chid,
            'adm_time': adm_time,
            'discharge_time': discharge_time,
        }

        return True, patinfo

    def read_readrslt_file(self):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'SC_PTE.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        if d['Label'] == 'Y' or d['Label'] == 'YY':
            pte_date = str(int(float(d['ExecYmd'])))
            pte_hours = "{:04d}".format(int(float(d['ExExecHm'])))
            pte_time = str2datetime(pte_date+' '+pte_hours)
            return True, pte_time
        else:
            return False, None

    def read_death_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.in_dir, self.chid, 'DEATH_20180824.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        death_time = str2datetime(d['ActDt'])

        return True, death_time

    def read_transfer_file(self) -> List[Transfer]:
        transfers = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'SEPSIS_KN_ICU_WARD_TRANSFER_SurrogateKey.pkl')
        if not os.path.exists(in_file):
            return transfers

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            icu_in_time = str2datetime(row['RgtDt'])
            icu_out_time = str2datetime(row['LstUpdDt'])

            if icu_in_time == NOT_CONVERTED or icu_out_time == NOT_CONVERTED:
                continue

            if row['Ward'] in ['ICUC', 'ICUD']:
                ward_type = 'SICU'
            else:
                ward_type = 'ICU'
            transfer = Transfer(dict(
                ward='ICU',
                ward_type=ward_type,
                in_time=icu_in_time,
                out_time=icu_out_time,
            ))
            transfers.append(transfer)

        return transfers

    def read_operation_file(self) -> List[Operation]:
        operations = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'SEPSIS_KN_OPERATIONHISTORY_SurrogateKey.pkl')
        if not os.path.exists(in_file):
            return operations

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            op_time = str2datetime(row['OpStrDateTime'])
            emergence = row['EmerYn']

            if op_time == NOT_CONVERTED:
                continue

            operation = Operation(dict(
                op_time=op_time,
                emergence=emergence
            ))
            operations.append(operation)

        return operations

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

    def read_dtr_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.dtr_dir, '{}.pkl'.format(self.chid))
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d: datetime = pickle.load(f)

        onset_time = str2datetime(d)

        return True, onset_time

    def read_pte_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.pte_dir, '{}.pkl'.format(self.chid))
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d: datetime = pickle.load(f)

        onset_time = str2datetime(d)

        return True, onset_time

    def read_cpr_file(self):
        cpr_times = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'CPR_20180824.pkl')
        if not os.path.exists(in_file):
            return cpr_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            cpr_time = str2datetime(row['MinTime'])
            cpr_times.append(cpr_time)

        return cpr_times

    def read_observation_file(self, feature_times):
        in_file_bases = [
            'SEPSIS_KN_TPR_SurrogateKey',
        ]

        for in_file_base in in_file_bases:
            in_file = os.path.join(self.arg.in_dir, self.chid, in_file_base)
            if not os.path.exists(in_file):
                continue

            with open(in_file, 'rb') as f:
                d = pickle.load(f)

            for feature_item in d:
                context_id = feature_item['OBSERVATION_CONCEPT_ID']
                if context_id not in FEATURE_FOR_DEFINE_PATIENT.OBSERVATION:
                    continue

                feature_code = FEATURE_FOR_DEFINE_PATIENT.OBSERVATION[context_id]
                feature_time = str2datetime(feature_item['OBSERVATION_DATETIME'])
                if feature_time == NOT_CONVERTED:
                    continue

                feature_value = feature_item['VALUE_AS_NUMBER']

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

                feature_times.add(feature_time)

        return feature_times

    def read_labresult_file(self, feature_times):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'SEPSIS_KN_Lab_SurrogateKey.pkl')
        if not os.path.exists(in_file):
            return feature_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['OrdNm']
            if context_id not in FEATURE_FOR_DEFINE_PATIENT.LAB:
                continue

            feature_code = FEATURE_FOR_DEFINE_PATIENT.LAB[context_id]

            feature_time = str2datetime(feature_item['preLabExamReptYmdHm'])
            if feature_time == NOT_CONVERTED:
                continue

            try:
                feature_value = float(feature_item['NmrcRslt'])
            except Exception:
                continue

            feature_times.add(feature_time)

        return feature_times

