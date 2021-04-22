import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial


def interpolate_patient(pid, features, use_features):
    if use_features:
        tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate', 'Fibrinogen',
                 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm',
                 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride',
                 'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
        pat_features = features.loc[features['pid'] == pid]

        features_flat = pat_features.groupby(['pid']).mean()
        features_flat = features_flat.join(pat_features.groupby(['pid']).std(), rsuffix='_std')

        # num Tests
        numTests = pd.DataFrame(pat_features[tests].notna().sum()).transpose()
        numTests.index = [pid]
        features_flat = features_flat.join(numTests, rsuffix='_nTests')

    else:
        pat_features = features.loc[features['pid'] == pid].interpolate(method='linear', axis=0, limit_direction='both')
        pat_features.fillna(0, inplace=True)

        features_flat = pat_features.set_index('pid').groupby(level=0).apply(
            lambda df: df.reset_index(drop=True)).unstack()
        features_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in features_flat.columns]

    return features_flat


def interpolate_per_patient(features, use_features):
    print("Start interpolate per patient:")

    # Parallelization
    pool = Pool(n_jobs)
    interpolated_features = pd.concat(pool.map(partial(interpolate_patient, features=features, use_features=use_features), features['pid'].unique()))

    print("Finished interpolation per patient")

    return interpolated_features


def drop_columns(features, drop_time_columns, use_features):
    if use_features:
        if drop_time_columns:
            features_dropped = features.drop('Time', axis=1)
        else:
            features_dropped = features
    else:
        # Drop Age/Time duplicates columns
        time_column_names = ['Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10',
                             'Time11', 'Time12']
        age_column_names = ['Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Age7', 'Age8', 'Age9', 'Age10', 'Age11', 'Age12']
        columns_to_drop = time_column_names + age_column_names if drop_time_columns else age_column_names
        features_dropped = features.drop(columns_to_drop, axis=1)

    return features_dropped


if __name__ == "__main__":
    #
    # read data (as a pd.DataFrame)
    #

    train_features = pd.read_csv('data/train_features.csv')
    test_features = pd.read_csv('data/test_features.csv')

    # how many cores to use, adapt to availability
    n_jobs = 7

    # use features (mean, std implemented) instead of concatenating
    use_features = True

    # If time is to be dropped
    drop_time = False

    #
    # Replace nans with interpolation per patient (or 0 if no value for this patient at all)
    #

    features_interpolated = interpolate_per_patient(train_features, use_features)
    train_features = drop_columns(features_interpolated, drop_time, use_features)

    test_features_interpolated = interpolate_per_patient(test_features, use_features)
    test_features = drop_columns(test_features_interpolated, drop_time, use_features)

    if use_features:
        train_features.to_csv('data/train_extracted_features.csv', index=True)
        test_features.to_csv('data/test_extracted_features.csv', index=True)
    else:
        train_features.to_csv('data/train_features_interpolated.csv', index=True)
        test_features.to_csv('data/test_features_interpolated.csv', index=True)
    print("interpolated datasets have been generated.")
