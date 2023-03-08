# coding: utf-8


class Profile:
    def __init__(self, d):
        self.pid = d['pid']
        self.chid = d['chid']
        self.adm_time = d['adm_time']
        self.discharge_time = d['discharge_time']
        self.event_time = d['event_time']
        self.event_yn = d['event_yn']
        self.pred_time = d['pred_time']

    def to_dict(self):
        return dict(
            pid=self.pid,
            chid=self.chid,
            adm_time=self.adm_time,
            discharge_time=self.discharge_time,
            event_time=self.event_time,
            event_yn=self.event_yn,
            pred_time=self.pred_time,
        )
