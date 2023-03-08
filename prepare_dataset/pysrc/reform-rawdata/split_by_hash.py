# coding: utf-8

import os
import csv
import argparse

from module_common import *
from common.util.hash_util import *
from common.util.process_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str)
parser.add_argument('--in_dir', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--chid_hash_size', type=int)
ARG = parser.parse_args()


def make_hash_dir():
    for i in range(ARG.chid_hash_size):
        chid_hash = get_chid_hash(i, ARG.chid_hash_size)
        out_chid_hash_dir = os.path.join(ARG.out_dir, chid_hash)
        os.mkdir(out_chid_hash_dir)


def split_data(file_meta):
    in_file = os.path.join(ARG.in_dir, file_meta.filename)
    with open(in_file, 'r', encoding='utf-8-sig') as in_fobj:
        reader = csv.DictReader(in_fobj)
        fieldnames = reader.fieldnames
        out_fobjs = dict()
        writers = dict()
        for i in range(ARG.chid_hash_size):
            chid_hash = get_chid_hash(i, ARG.chid_hash_size)
            out_chid_hash_dir = os.path.join(ARG.out_dir, chid_hash)
            out_file = os.path.join(out_chid_hash_dir, file_meta.filename)
            out_fobj = open(out_file, 'w', encoding='utf-8')
            out_fobjs[chid_hash] = out_fobj
            writers[chid_hash] = csv.writer(out_fobj)
            writers[chid_hash].writerow(fieldnames)

        for row in tqdm(reader, desc='reading file: {}'.format(file_meta.filename)):
            chid = row[file_meta.chid_field]
            if chid == '' or chid == None:
                continue

            if ARG.dataset in ['ILSAN']:
                chid_int = int(chid, 16)
            elif ARG.dataset in ['SVRC', 'SVRCLIVE', 'MIMIC', 'MIMICED', 'SVRCKN']:
                try:
                    chid_int = int(chid)
                except:
                    continue
            else:
                raise AssertionError()

            chid_hash = get_chid_hash(chid_int, ARG.chid_hash_size)
            row_values = [row[x] for x in fieldnames]
            writers[chid_hash].writerow(row_values)

        for i in range(ARG.chid_hash_size):
            chid_hash = get_chid_hash(i, ARG.chid_hash_size)
            out_fobjs[chid_hash].close()


def main():
    make_hash_dir()
    file_meta_list = RAWFILE_META_MAP[ARG.dataset]
    run_multi_process(split_data, file_meta_list, n_processes=min(len(file_meta_list), 10))


if __name__ == '__main__':
    main()
