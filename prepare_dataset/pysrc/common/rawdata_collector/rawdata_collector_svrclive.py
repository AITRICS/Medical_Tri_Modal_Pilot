# coding: utf-8

import os
import csv
import pickle
from datetime import timedelta

from common.util.datetime_util import *
from typing import *
from common.feature_table.feature_table_svrclive import *


class Transfer:
    def __init__(self, d=None):
        if d:
            self.ward = d['ward']
            self.in_time = d['in_time']
            self.out_time = d['out_time']

    def __repr__(self):
        return '{} {} ~ {}'.format(self.ward, self.in_time, self.out_time)


class RawdataCollector:
    def __init__(self, arg, chid):
        self.arg = arg
        self.chid = chid

    def read_patient_file(self) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.in_dir, self.chid, 'emr_patient.pkl')
        with open(in_file, 'rb') as f:
            d = pickle.load(f)[0]

        adm_time = d['adm_date']
        discharge_time = d['discharge_date']

        if adm_time == 'NULL' or discharge_time == 'NULL':
            return False, None

        adm_time = str2datetime(adm_time)
        discharge_time = str2datetime(discharge_time)
        discharge_time += timedelta(days=1)

        emr_id = d['emr_id']

        patinfo = {
            'chid': self.chid,
            'uid': emr_id,
            'adm_time': adm_time,
            'discharge_time': discharge_time,
        }

        return True, patinfo

    def read_death_file(self, patinfo: dict) -> Tuple[bool, Any]:
        in_file = os.path.join(self.arg.rawdata_dir, 'deathtime_20190729.csv')
        uid_deaths = dict()
        with open(in_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row['UnitNo'].strip()
                dead_ymd = row['DeadYmd'].strip()
                dead_hms = row['DeadHms'].strip()

                death_time = str2datetime('{}{}00'.format(dead_ymd, dead_hms))
                if death_time == NOT_CONVERTED:
                    continue

                uid_deaths[uid] = death_time

        uid = patinfo['uid']
        if uid in uid_deaths:
            return True, uid_deaths[uid]
        else:
            return False, None

    def read_transfer_file(self) -> List[Transfer]:
        transfers = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'emr_encounter.pkl')
        if not os.path.exists(in_file):
            return transfers

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        entrances = list()
        for row in d:
            in_time = str2datetime(row['start_dt'])
            if in_time == NOT_CONVERTED:
                continue

            encounter_type = row['type']
            if encounter_type == 'admission':
                continue

            dest = row['destination_text']
            entrance = dict(
                in_time=in_time,
                ward=dest,
            )
            entrances.append(entrance)

        for entrance in entrances:
            if entrance['ward'] in {'신생아집중치료실', 'NICU', 'PICU'}:
                return transfers

        icu_wards = {'중환자실', 'CCU', 'ICU', 'ICUA', 'ICUB', 'NCU', 'NCUA'}
        entrances = list(map(lambda x: {'in_time': x['in_time'],
                                        'ward': 'ICU' if x['ward'] in icu_wards else 'GW'},
                             entrances))

        entrances = sorted(entrances, key=lambda x: x['in_time'])
        curr_ward = None
        curr_in_time = None
        transfers = list()
        for entrance in entrances:
            ward = entrance['ward']
            in_time = entrance['in_time']
            if curr_ward:
                if curr_ward == 'ICU' and ward == 'GW':
                    transfer = Transfer(dict(
                        ward='ICU',
                        in_time=curr_in_time,
                        out_time=in_time,
                    ))
                    transfers.append(transfer)
                    curr_ward = ward
                    curr_in_time = in_time
                elif curr_ward == 'ICU' and ward == 'ICU':
                    continue
                elif curr_ward == 'GW' and ward == 'ICU':
                    transfer = Transfer(dict(
                        ward='GW',
                        in_time=curr_in_time,
                        out_time=in_time,
                    ))
                    transfers.append(transfer)
                    curr_ward = ward
                    curr_in_time = in_time
                elif curr_ward == 'GW' and ward == 'GW':
                    continue
            else:
                curr_ward = ward
                curr_in_time = in_time
        if curr_ward:
            transfer = Transfer(dict(
                ward=curr_ward,
                in_time=curr_in_time,
                out_time=str2datetime('99991231'),
            ))
            transfers.append(transfer)

        return transfers

    def read_cpr_file(self) -> List[datetime]:
        cpr_times = list()

        in_file = os.path.join(self.arg.in_dir, self.chid, 'emr_nursingrecord.pkl')
        if not os.path.exists(in_file):
            return cpr_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for row in d:
            action_time = str2datetime(row['action_dt'])
            action_code = str2datetime(row['action_code'])
            if action_code in {'NA000721', 'NA000727'}:
                cpr_times.append(action_time)

        return cpr_times

    def read_sepsis_file(self):
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

    def read_observation_file(self, feature_times):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'emr_observation.pkl')
        if not os.path.exists(in_file):
            return feature_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['context_id']
            if context_id not in FEATURE_FOR_DEFINE_PATIENT.OBSERVATION:
                continue

            feature_code = FEATURE_FOR_DEFINE_PATIENT.OBSERVATION[context_id]
            feature_time = str2datetime(feature_item['observe_dt'])
            try:
                feature_value = float(feature_item['value'])
            except Exception:
                continue

            feature_times.add(feature_time)

        return feature_times

    def read_labresult_file(self, feature_times):
        in_file = os.path.join(self.arg.in_dir, self.chid, 'emr_labresult.pkl')
        if not os.path.exists(in_file):
            return feature_times

        with open(in_file, 'rb') as f:
            d = pickle.load(f)

        for feature_item in d:
            context_id = feature_item['context_id']
            if context_id not in FEATURE_FOR_DEFINE_PATIENT.LAB:
                continue

            feature_code = FEATURE_FOR_DEFINE_PATIENT.LAB[context_id]
            feature_time = str2datetime(feature_item['result_dt'])
            try:
                feature_value = float(feature_item['value'])
            except Exception:
                continue

            feature_times.add(feature_time)

        return feature_times
