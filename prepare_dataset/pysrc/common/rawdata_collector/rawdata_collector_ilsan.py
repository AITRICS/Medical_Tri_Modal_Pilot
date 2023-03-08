# coding: utf-8

import os
import pickle
from datetime import timedelta

from common.util.datetime_util import *
from typing import *
from common.feature_table.feature_table_ilsan import *


class Transfer:
    def __init__(self, d=None):
        if d:
            self.ward = d['ward']
            self.ward_type = d['ward_type']
            self.in_time = d['in_time']
            self.out_time = d['out_time']

    def __repr__(self):
        return '{} {} {} ~ {}'.format(self.ward, self.ward_type, self.in_time, self.out_time)


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
        in_file = os.path.join(self.arg.in_dir, self.chid, 'PERSON.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        adm_time = d['AdmiYmd']
        discharge_time = d['OtrmYmd']

        adm_time = str2datetime(adm_time)
        discharge_time = str2datetime(discharge_time)
        discharge_time += timedelta(days=1)

        profile = {
            'chid': self.chid,
            'adm_time': adm_time,
            'discharge_time': discharge_time,
        }

        return True, profile

    def read_readrslt_file(self):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'IS_PTE_ALL.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            ds = pickle.load(f)

        for d in ds:
            pte_times = list()
            if d['Label'] == 'Y' or d['Label'] == 'YY':
                pte_date = str(int(float(d['ExecYmd'])))
                pte_hours = "{:04d}".format(int(float(d['ExExecHm'])))
                pte_time = str2datetime(pte_date + ' ' + pte_hours)
                pte_times.append(pte_time)

        if len(pte_times) > 0:
            pte_times = sorted(pte_times)
            return True, pte_times[0]

        return False, None

    def read_death_file(self):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'PERSON.pkl')
        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        death_time = d['DeathDateTime']
        if death_time != '':
            death_time = str2datetime(death_time)
            return True, death_time

        return False, None

    def read_observation_file(self, feature_times):
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

                feature_value = feature_item['OBSERVATION_SOURCE_VALUE']

                if feature_code == 'GCS':
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

                feature_times.add(feature_time)

        return feature_times

    def read_labresult_file(self, feature_times):
        in_file_bases = [
            'LAB_201701.pkl',
            'LAB_201704.pkl',
            'LAB_201707.pkl',
            'LAB_201710.pkl',
        ]

        for in_file_base in in_file_bases:
            in_file = os.path.join(self.arg.in_dir, self.chid, in_file_base)
            if not os.path.exists(in_file):
                continue

            with open(in_file, 'rb') as f:
                d = pickle.load(f)

            for feature_item in d:
                context_id = feature_item['preLabOrdNm']
                if context_id not in FEATURE_FOR_DEFINE_PATIENT.LAB:
                    continue

                feature_code = FEATURE_FOR_DEFINE_PATIENT.LAB[context_id]

                feature_time = str2datetime(feature_item['preLabOrdExecYmdHm'])
                if feature_time == NOT_CONVERTED:
                    continue

                try:
                    feature_value = float(feature_item['preLabNmrcRslt'])
                except Exception:
                    continue

                feature_times.add(feature_time)

        return feature_times

    def read_transfer_file(self):
        transfers = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'ICU_WARD_TRANSFER_20181129.pkl')
        if not os.path.exists(in_file):
            return True, transfers

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        entrances = list()
        for row in d:
            in_date = row['InDate']
            in_time = row['RgtDateTime']

            in_time = str2datetime('{}{}00'.format(in_date, in_time))
            if in_time == NOT_CONVERTED:
                continue

            ward_str = row['Ward']
            if ward_str in {'WCAT', 'NICU', 'RMI', 'DRM', 'EMC', 'NUR'}:
                return False, transfers
            elif ward_str in {'SICU', 'MICU', 'CCU'}:
                ward = 'ICU'
            else:
                ward = 'GW'

            entrance = dict(
                in_time=in_time,
                ward=ward,
                ward_type=ward_str,
            )
            entrances.append(entrance)

        entrances = sorted(entrances, key=lambda x: x['in_time'])
        curr_ward = None
        curr_ward_type = None
        curr_in_time = None
        transfers = list()
        for entrance in entrances:
            ward = entrance['ward']
            in_time = entrance['in_time']
            if curr_ward:
                if curr_ward == 'ICU' and ward == 'GW':
                    transfer = Transfer(dict(
                        ward='ICU',
                        ward_type=curr_ward_type,
                        in_time=curr_in_time,
                        out_time=in_time,
                    ))
                    transfers.append(transfer)
                    curr_ward = ward
                    curr_ward_type = entrance['ward_type']
                    curr_in_time = in_time
                elif curr_ward == 'ICU' and ward == 'ICU':
                    continue
                elif curr_ward == 'GW' and ward == 'ICU':
                    transfer = Transfer(dict(
                        ward='GW',
                        ward_type='GW',
                        in_time=curr_in_time,
                        out_time=in_time,
                    ))
                    transfers.append(transfer)
                    curr_ward = ward
                    curr_ward_type = entrance['ward_type']
                    curr_in_time = in_time
                elif curr_ward == 'GW' and ward == 'GW':
                    continue
            else:
                curr_ward = ward
                curr_ward_type = entrance['ward_type']
                curr_in_time = in_time
        if curr_ward:
            transfer = Transfer(dict(
                ward=curr_ward,
                ward_type=curr_ward,
                in_time=curr_in_time,
                out_time=str2datetime('99991231'),
            ))
            transfers.append(transfer)

        return True, transfers

    def read_operation_file(self) -> List[Operation]:
        operations = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'OPERATION_HISTORY.pkl')
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
        if self.arg.ward == 'GW':
            in_file = os.path.join(self.arg.sepsis_dir, 'GW_{}.pkl'.format(self.chid))
        elif self.arg.ward == 'ICU':
            in_file = os.path.join(self.arg.sepsis_dir, 'ICU_{}.pkl'.format(self.chid))

        if not os.path.exists(in_file):
            return False, None

        with open(in_file, 'rb') as f:
            d: datetime = pickle.load(f)

        onset_time = str2datetime(d)

        return True, onset_time

    def read_aki_file(self) -> Tuple[bool, Any]:
        if self.arg.ward == 'GW':
            in_file = os.path.join(self.arg.aki_dir, 'GW_{}.pkl'.format(self.chid))
        elif self.arg.ward == 'ICU':
            in_file = os.path.join(self.arg.aki_dir, 'ICU_{}.pkl'.format(self.chid))
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

    def read_cpr_file(self) -> List[datetime]:
        cpr_times = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'PERSONNP000006.pkl')
        if not os.path.exists(in_file):
            return cpr_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            term_na = row['Term_NA']
            action_time = str2datetime(row['ActDt'])

            if term_na in {'심장 압박을 시행함 ', '심장 압박을 시행함'}:
                cpr_times.append(action_time)

        return cpr_times
