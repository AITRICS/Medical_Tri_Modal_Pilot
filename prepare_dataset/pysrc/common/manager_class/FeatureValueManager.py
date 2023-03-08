# coding: utf-8


class FeatureValueManager:
    def __init__(self):
        self.value_ranges = dict(
            AGE=(18, 90),
            PULSE=(0, 300),
            RESP=(0, 120),
            SBP=(0, 300),
            DBP=(0, 300),
            TEMP=(25, 50),
            SpO2=(0, 100),
            GCS=(3, 15),

            BILIRUBIN=(0, 75),
            LACTATE=(0, 20),
            pH=(0, 14),
            SODIUM=(0, 500),
            POTASSIUM=(0, 15),
            CREATININE=(0, 20),
            HEMATOCRIT=(0, 100),
            WBC=(0, 100),
            HCO3=(0, 100),
            PLATELET=(0, 1000),
            CRP=(0,900),
            DDIMER=(0,100)
            # DNI=(0, 100),
        )

    def is_in_range(self, feature_key, feature_value):
        if feature_key not in self.value_ranges:
            raise AssertionError(feature_key)

        value_min, value_max = self.value_ranges[feature_key]
        return value_min <= feature_value <= value_max
