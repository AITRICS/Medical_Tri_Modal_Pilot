# coding: utf-8


def get_chid_hash(i: int, chid_hash_size: int):
    chid_hash_int = int(i % chid_hash_size)
    return 'hash-{:06}'.format(chid_hash_int)
