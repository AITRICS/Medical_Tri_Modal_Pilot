# coding: utf-8

from common.feature_table.feature_code import FEATURE_CODE


class FEATURE_FOR_DEFINE_PATIENT:
    OBSERVATION = dict([
        ('220045', FEATURE_CODE.PULSE),
        ('211', FEATURE_CODE.PULSE),
        ('220210', FEATURE_CODE.RESP),
        ('618', FEATURE_CODE.RESP),
        ('223762', FEATURE_CODE.TEMP_C),
        ('676', FEATURE_CODE.TEMP_C),
        ('677', FEATURE_CODE.TEMP_C),
        ('223761', FEATURE_CODE.TEMP_F),
        ('678', FEATURE_CODE.TEMP_F),
        ('679', FEATURE_CODE.TEMP_F),
        ('220050', FEATURE_CODE.SBP),
        ('220179', FEATURE_CODE.SBP),
        ('225309', FEATURE_CODE.SBP),
        ('51', FEATURE_CODE.SBP),
        ('455', FEATURE_CODE.SBP),
        ('220051', FEATURE_CODE.DBP),
        ('220180', FEATURE_CODE.DBP),
        ('225310', FEATURE_CODE.DBP),
        ('8368', FEATURE_CODE.DBP),
        ('8441', FEATURE_CODE.DBP),
        
        ('220277', FEATURE_CODE.SpO2),
        ('646', FEATURE_CODE.SpO2),

        ('198', FEATURE_CODE.GCS),
        ('220739', FEATURE_CODE.GCS_EYE),
        ('223900', FEATURE_CODE.GCS_VER),
        ('223901', FEATURE_CODE.GCS_MOT),

        # ('2001100029', FEATURE_CODE.MBP),
    ])

    LAB = dict([
        ('50885', FEATURE_CODE.BILIRUBIN),
        ('50912', FEATURE_CODE.CREATININE),
        ('50813', FEATURE_CODE.LACTATE),
        ('51265', FEATURE_CODE.PLATELET),
        # ('Delta neutrophil 1[Whole blood]', FEATURE_CODE.DNI),
    ])


class FEATURE_FOR_EXTRACT_FEATURE:
    OBSERVATION = dict([
        ('220045', FEATURE_CODE.PULSE),
        ('211', FEATURE_CODE.PULSE),
        ('220210', FEATURE_CODE.RESP),
        ('618', FEATURE_CODE.RESP),
        ('223762', FEATURE_CODE.TEMP_C),
        ('676', FEATURE_CODE.TEMP_C),
        ('677', FEATURE_CODE.TEMP_C),
        ('223761', FEATURE_CODE.TEMP_F),
        ('678', FEATURE_CODE.TEMP_F),
        ('679', FEATURE_CODE.TEMP_F),
        ('220050', FEATURE_CODE.SBP),
        ('220179', FEATURE_CODE.SBP),
        ('225309', FEATURE_CODE.SBP),
        ('51', FEATURE_CODE.SBP),
        ('455', FEATURE_CODE.SBP),
        ('220051', FEATURE_CODE.DBP),
        ('220180', FEATURE_CODE.DBP),
        ('225310', FEATURE_CODE.DBP),
        ('8368', FEATURE_CODE.DBP),
        ('8441', FEATURE_CODE.DBP),
        
        ('220277', FEATURE_CODE.SpO2),
        ('646', FEATURE_CODE.SpO2),

        ('198', FEATURE_CODE.GCS),
        ('220739', FEATURE_CODE.GCS_EYE),
        ('223900', FEATURE_CODE.GCS_VER),
        ('223901', FEATURE_CODE.GCS_MOT),
        # ('2001100029', FEATURE_CODE.MBP),
        # ('2001100125', FEATURE_CODE.FiO2),
        # ('2001100086', FEATURE_CODE.O2),
    ])

    LAB = dict([
        ('50885', FEATURE_CODE.BILIRUBIN),                          #
        ('50813', FEATURE_CODE.LACTATE),                                #
        # ('Lactate[Venous Whole blood]', FEATURE_CODE.LACTATE),
        ('50820', FEATURE_CODE.pH),                                     #
        ('50831', FEATURE_CODE.pH),                                 #
        ('50983', FEATURE_CODE.SODIUM),                             #
        ('50971', FEATURE_CODE.POTASSIUM),                          #
        ('50912', FEATURE_CODE.CREATININE),                         #
        ('51221', FEATURE_CODE.HEMATOCRIT),                         #
        ('51301', FEATURE_CODE.WBC),                                   #
        ('50882', FEATURE_CODE.HCO3),                               #
        # ('Delta neutrophil 1[Whole blood]', FEATURE_CODE.DNI),
        ('50889', FEATURE_CODE.CRP),
        ('51265', FEATURE_CODE.PLATELET),                           #
    ])
