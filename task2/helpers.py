# Test second approach here

import pandas as pd
import numpy as np


def inspect_nan_statistics_per_feature(data):
    """
    count number of NaNs in each column for each patient. Summarize and print stats of those counts.
    """
    n_timesteps = 12
    n_patients = int(data.shape[0] / n_timesteps)
    print("There are " + str(n_patients) + " patients in this dataframe")
    data = data.notna()  # boolean array (1 if value is not NaN, 0 if Nan)
    nan_count_per_patient = np.zeros((n_patients, data.shape[1]))

    for i in range(n_patients):
        patient_data = data.iloc[n_timesteps*i:(n_timesteps*(i+1)), :]  # select rows corresponding to patient
        nan_count_per_patient[i] = np.sum(patient_data, axis=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        nan_count_per_patient = pd.DataFrame(nan_count_per_patient)
        nan_count_per_patient.columns = train_features.columns
        stats = nan_count_per_patient.describe()
        print(stats)


if __name__ == "__main__":
    train_features = pd.read_csv('data/train_features.csv')
    inspect_nan_statistics_per_feature(train_features)
