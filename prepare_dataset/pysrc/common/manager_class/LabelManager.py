# coding: utf-8

import os
import pickle

from common.object_class.EventLabel import EventLabel


class LabelManager:
    def __init__(self, filename):
        self.filename = filename

    def dump_event_labels(self, labels: list):
        labels = list(map(lambda x: x.to_dict(), labels))
        with open(self.filename, 'wb') as f:
            pickle.dump(labels, f)

    def load_label_items(self):
        with open(self.filename, 'rb') as f:
            l = pickle.load(f)
            l = list(map(lambda x: EventLabel(x), l))
            l = {x.pid: x for x in l}

            return l
