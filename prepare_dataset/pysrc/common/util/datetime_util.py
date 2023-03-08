# coding: utf-8

from datetime import datetime

NOT_CONVERTED = 'NOT_CONVERTED'


def str2datetime(s):
    def _convert(_s, _dformat):
        try:
            converted_dt = datetime.strptime(_s, _dformat)
        except Exception:
            return NOT_CONVERTED

        return converted_dt

    if isinstance(s, datetime):
        return s

    dformats = [
        '%Y-%m-%d %p %I:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y%m%d',
        '%Y-%m-%d +%H:%M',
        '%Y%m%d %H%M',
        '%Y%m%d%H%M%S',
        ]

    s = s.split('.')[0]
    s = s.replace('오전', 'am').replace('오후', 'pm')
    for dformat in dformats:
        dt = _convert(s, dformat)
        if dt != NOT_CONVERTED:
            return dt

    return NOT_CONVERTED


def calc_year_difference(t_min, t_max):
    return (t_max - t_min).total_seconds() / (60*60*24*365.25)


def datetime_to_hours(time):
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    return (year * 8760) + (month * 730) + (day * 24) + hour + (minute / float(60))
