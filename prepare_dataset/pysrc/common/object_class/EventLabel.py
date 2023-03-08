# coding: utf-8


class EventLabel:
    def __init__(self, d):
        self.pid = d['pid']
        self.label = d['label']
        self.event_dt = d['event_dt']
        self.adm_dt = d['adm_dt']
        self.discharge_dt = d['discharge_dt']
        self.death_dt = d['death_dt']

    def to_dict(self):
        return dict(
            pid=self.pid,
            label=self.label,
            event_dt=self.event_dt,
            adm_dt=self.adm_dt,
            discharge_dt=self.discharge_dt,
            death_dt=self.death_dt,
        )
