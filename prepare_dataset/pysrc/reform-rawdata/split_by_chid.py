# coding: utf-8

import os
import csv
import argparse

import pickle

from module_common import *
from common.util.hash_util import *
from common.util.process_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset'         , type=str)
parser.add_argument('--out_dir'         , type=str)
parser.add_argument('--in_dir'          , type=str)
parser.add_argument('--chid_hash_size'  , type=int)
ARG = parser.parse_args()


def split_data(chid_hash):
    in_hash_dir = os.path.join(ARG.in_dir, chid_hash)
    for file_meta in RAWFILE_META_MAP[ARG.dataset]:
        records = dict()
        in_file = os.path.join(in_hash_dir, file_meta.filename)
        with open(in_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                chid = row[file_meta.chid_field]
                records[chid] = records.get(chid, list())
                records[chid].append({x: row[x] for x in reader.fieldnames})

        for chid in records:
            out_chid_dir = os.path.join(ARG.out_dir, chid)
            if not os.path.exists(out_chid_dir):
                os.mkdir(out_chid_dir)
            out_file = os.path.join(out_chid_dir, '{}.pkl'.format(file_meta.filename[:-4]))
            with open(out_file, 'wb') as f:
                pickle.dump(records[chid], f)

# def split_data(chid_hash):
#     in_hash_dir = os.path.join(ARG.in_dir, chid_hash)
#     for file_meta in RAWFILE_META_MAP[ARG.dataset]:
#         records = dict()
#         in_file = os.path.join(in_hash_dir, file_meta.filename)
#         with open(in_file, 'r', encoding='utf-8-sig') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 chid = row[file_meta.chid_field]
#                 records[chid] = records.get(chid, list())
#                 records[chid].append({x: row[x] for x in reader.fieldnames})

#         for chid in records:
#             out_chid_dir = os.path.join(ARG.out_dir, chid)
#             if not os.path.exists(out_chid_dir):
#                 os.mkdir(out_chid_dir)
#             out_file = os.path.join(out_chid_dir, '{}.csv'.format(file_meta.filename[:-4]))
#             if len(records[chid]) > 0:
#                 fields = list(records[chid][0].keys())
#                 writer = csv.DictWriter(open(out_file, 'w', encoding='utf-8-sig'), fieldnames=fields)
#                 writer.writeheader()
#                 for i, row in enumerate(records[chid]):
#                     writer.writerow(row)
                


def main():
    chid_hash_list = [get_chid_hash(i, ARG.chid_hash_size) for i in range(ARG.chid_hash_size)]
    run_multi_process(split_data, chid_hash_list)


if __name__ == '__main__':
    main()
