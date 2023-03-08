# coding: utf-8


def is_valid_feature_value(feature_key, v):
    if feature_key == 'PULSE':
        return 0 <= v <= 300
    elif feature_key == 'RESP':
        return 0 <= v <= 120
    elif feature_key == 'SBP':
        return 0 <= v <= 300
    elif feature_key == 'DBP':
        return 0 <= v <= 300
    elif feature_key == 'TEMP':
        return 25 <= v <= 50
    elif feature_key == 'GCS':
        return 3 <= v <= 15
    elif feature_key == 'FiO2':
        return 0 <= v <= 1
    elif feature_key == 'O2':
        return 1 <= v <= 10
    elif feature_key == 'SpO2':
        return 0 <= v <= 100
    elif feature_key == 'pO2':
        return True
    elif feature_key == 'PLATELET':
        return 0 <= v <= 1000
    elif feature_key == 'BILIRUBIN':
        return 0 <= v <= 75
    elif feature_key == 'MBP':
        return True
    elif feature_key == 'NORP':
        return True
    elif feature_key == 'DOPA':
        return True
    elif feature_key == 'DOBU':
        return True
    elif feature_key == 'VASO':
        return True
    elif feature_key == 'EPIN':
        return True
    elif feature_key == 'URINE':
        return True
    elif feature_key == 'CREATININE':
        return 0 <= v <= 20
    elif feature_key == 'SaO2':
        return True
    elif feature_key == 'GCS_EYE':
        return True
    elif feature_key == 'GCS_MOT':
        return True
    elif feature_key == 'GCS_VER':
        return True
    elif feature_key == 'GCS_TOTAL':
        return 3 <= v <= 15
    elif feature_key == 'SODIUM':
        return 0 <= v <= 500
    elif feature_key == 'pH':
        return 0 <= v <= 14
    elif feature_key == 'POTASSIUM':
        return 0 <= v <= 15
    elif feature_key == 'WBC':
        return 0 <= v <= 100
    elif feature_key == 'HCO3':
        return 0 <= v <= 100
    elif feature_key == 'HEMATOCRIT':
        return 0 <= v <= 100

    else:
        raise AssertionError('{} {}'.format(feature_key, v))