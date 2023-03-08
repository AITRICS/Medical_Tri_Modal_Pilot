# coding: utf-8

from scipy.stats import sem, t
from scipy import mean


def calc_confidence_interval(data, confidence=0.95):
    confidence = confidence
    print(confidence)
    # confidence = 0.975

    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h

    ci_result = dict(
        COUNT=n,
        MEAN=m,
        STD_ERR=std_err,
        CI_ERR=h,
        START=start,
        END=end,
    )

    return ci_result
