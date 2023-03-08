# coding: utf-8


class RawfileMeta:
    def __init__(self, filename, chid_field):
        self.filename = filename
        self.chid_field = chid_field


RAWFILE_META_MAP = {
    'SVRC': [
        RawfileMeta('OBSERVATION_VS_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('OBSERVATION_VS2014_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('OBSERVATION_VS2015_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('OBSERVATION_VS2016_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('OBSERVATION_VS2017_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('Lab2016_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('SEPSIS_LAB_New_SurrogateKey_utf8.csv', 'ChosNo'),
        RawfileMeta('DEATH_20180824.csv', 'ChosNo'),
        RawfileMeta('PERSON_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('TIME_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('SEPSIS_DRUG_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('SEPSIS_Frm_CRRT_respiration_SurrogateKey_utf8.csv', 'ChosNo'),
        RawfileMeta('SEPSIS_DRUG_New_SurrogateKey_utf8.csv', 'ChosNo'),
        RawfileMeta('SEPSIS_Frm_hemodialysis_SurrogateKey_utf8.csv', 'ChosNo'),
        RawfileMeta('SEPSIS_PERSON_HtWght_SurrogateKey_utf8.csv', 'ChosNo'),
        RawfileMeta('CPR_20180824.csv', 'ChosNo'),
        RawfileMeta('섭취배설량.csv', 'ChosNo'),
        RawfileMeta('OPList_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('퇴원요약.csv', 'ChosNo'),
        RawfileMeta('수술정보.csv', 'ChosNo'),
        RawfileMeta('입원기록.csv', 'ChosNo'),
        RawfileMeta('처치간호정보.csv', 'ChosNo'),
        RawfileMeta('간호정보조사.csv', 'ChosNo'),
        RawfileMeta('기능검사_결과_서식형_impression.csv', 'ChosNo'),
        RawfileMeta('Diagnosis_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('OPList_SurrogateKey.csv', 'ChosNo'),
        RawfileMeta('DICOM_Info.csv', 'ChosNo'),  # X-ray Image Data
    ],

    'MIMIC': [
        RawfileMeta('ADMISSIONS.csv', 'HADM_ID'),
        RawfileMeta('CHARTEVENTS.csv', 'HADM_ID'),
        RawfileMeta('DATETIMEEVENTS.csv', 'HADM_ID'),
        RawfileMeta('DIAGNOSES_ICD.csv', 'HADM_ID'),
        RawfileMeta('DRGCODES.csv', 'HADM_ID'),
        RawfileMeta('EMAR.csv', 'HADM_ID'),
        # RawfileMeta('EMAR_DETAIL.csv', 'HADM_ID'),
        RawfileMeta('HCPCSEVENTS.csv', 'HADM_ID'),
        RawfileMeta('ICUSTAYS.csv', 'HADM_ID'),
        RawfileMeta('INPUTEVENTS.csv', 'HADM_ID'),
        RawfileMeta('LABEVENTS.csv', 'HADM_ID'),
        RawfileMeta('MICROBIOLOGYEVENTS.csv', 'HADM_ID'),
        RawfileMeta('OUTPUTEVENTS.csv', 'HADM_ID'),
        RawfileMeta('PHARMACY.csv', 'HADM_ID'),
        RawfileMeta('POE.csv', 'HADM_ID'),
        # RawfileMeta('POE_DETAIL.csv', 'HADM_ID'),
        RawfileMeta('PRESCRIPTIONS.csv', 'HADM_ID'),
        RawfileMeta('PROCEDUREEVENTS.csv', 'HADM_ID'),
        RawfileMeta('PROCEDURES_ICD.csv', 'HADM_ID'),
        RawfileMeta('SERVICES.csv', 'HADM_ID'),
        RawfileMeta('TRANSFERS.csv', 'HADM_ID'),

    ],

    'MIMICED': [
        RawfileMeta('diagnosis.csv', 'stay_id'),
        RawfileMeta('edstays.csv', 'stay_id'),
        RawfileMeta('medrecon.csv', 'stay_id'),
        RawfileMeta('pyxis.csv', 'stay_id'),
        RawfileMeta('triage.csv', 'stay_id'),
        RawfileMeta('vitalsign.csv', 'stay_id'),
    ],
}
