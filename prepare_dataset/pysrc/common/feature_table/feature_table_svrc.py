# coding: utf-8

from common.feature_table.feature_code import FEATURE_CODE


class FEATURE_FOR_DEFINE_PATIENT:
    OBSERVATION = dict([
        ('1000500001', FEATURE_CODE.PULSE),
        ('1000600001', FEATURE_CODE.RESP),
        ('1000800001', FEATURE_CODE.SBP),
        ('1000900001', FEATURE_CODE.DBP),
        ('1000700001', FEATURE_CODE.TEMP),
        ('2001100049', FEATURE_CODE.SpO2),
        ('5002500001', FEATURE_CODE.GCS),
        # ('2001100029', FEATURE_CODE.MBP),
    ])

    LAB = dict([
        ('T. Bilirubin[Serum]', FEATURE_CODE.BILIRUBIN),
        ('Creatinine[Serum]', FEATURE_CODE.CREATININE),
        ('Lactate[Arterial Whole blood]', FEATURE_CODE.LACTATE),
        ('Delta neutrophil 1[Whole blood]', FEATURE_CODE.DNI),
    ])

    LAB_NEW = dict([
        ('PLT Count', FEATURE_CODE.PLATELET),
    ])


class FEATURE_FOR_EXTRACT_FEATURE:
    OBSERVATION = dict([
        ('1000500001', FEATURE_CODE.PULSE),
        ('1000600001', FEATURE_CODE.RESP),
        ('1000800001', FEATURE_CODE.SBP),
        ('1000900001', FEATURE_CODE.DBP),
        ('1000700001', FEATURE_CODE.TEMP),
        ('2001100049', FEATURE_CODE.SpO2),
        ('5002500001', FEATURE_CODE.GCS),
        ('2001100029', FEATURE_CODE.MBP),
        ('2001100125', FEATURE_CODE.FiO2),
        ('2001100086', FEATURE_CODE.O2),
        ('4001600001', FEATURE_CODE.URINE),
        ('4001600002', FEATURE_CODE.URINE),
        ('4001600003', FEATURE_CODE.URINE),
        ('4001600004', FEATURE_CODE.URINE),
        ('4001600005', FEATURE_CODE.URINE),
        ('4001600006', FEATURE_CODE.URINE),
        ('4001600007', FEATURE_CODE.URINE),
        ('4001600008', FEATURE_CODE.URINE),
        ('4001600009', FEATURE_CODE.URINE),
        ('4001600010', FEATURE_CODE.URINE),
        ('4001600011', FEATURE_CODE.URINE),
        ('4001600012', FEATURE_CODE.URINE),
    ])

    LAB = dict([
        ('T. Bilirubin[Serum]', FEATURE_CODE.BILIRUBIN),
        ('Lactate[Arterial Whole blood]', FEATURE_CODE.LACTATE),
        ('Lactate[Venous Whole blood]', FEATURE_CODE.LACTATE),
        ('pH[Arterial Whole blood]', FEATURE_CODE.pH),
        ('Na[Serum]', FEATURE_CODE.SODIUM),
        ('K[Serum]', FEATURE_CODE.POTASSIUM),
        ('Creatinine[Serum]', FEATURE_CODE.CREATININE),
        ('Hct[Arterial Whole blood]', FEATURE_CODE.HEMATOCRIT),
        ('Hct[Whole blood]', FEATURE_CODE.HEMATOCRIT),
        ('WBC COUNT[Whole blood]', FEATURE_CODE.WBC),
        ('HCO3-[Arterial Whole blood]', FEATURE_CODE.HCO3),
        # ('Delta neutrophil 1[Whole blood]', FEATURE_CODE.DNI),
        ('CRP (C-Reactive Protein)[Serum]', FEATURE_CODE.CRP),
        ('D-Dimer정량[Plasma]', FEATURE_CODE.DDIMER),

    ])

    LAB_NEW = dict([
        ('PLT Count', FEATURE_CODE.PLATELET),
    ])
