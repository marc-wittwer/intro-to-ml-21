import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial


def interpolate_patient(pid, features):
    pat_features = features.loc[features['pid'] == pid].interpolate(method='linear', axis=0, limit_direction='both')
    pat_features.fillna(0, inplace=True)

    features_flat = pat_features.set_index('pid').groupby(level=0).apply(
        lambda df: df.reset_index(drop=True)).unstack()
    features_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in features_flat.columns]
    return features_flat


def interpolate_per_patient(features):
    print("Start interpolate per patient:")

    # Parallelization
    pool = Pool(n_jobs)
    interpolated_features = pd.concat(pool.map(partial(interpolate_patient, features=features), features['pid'].unique()))

    print("Finished interpolation per patient")

    return interpolated_features


def drop_columns(features, drop_time_columns):
    # Drop Age/Time duplicates columns
    time_column_names = ['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10',
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
    n_jobs = 3

    #
    # Replace nans with interpolation per patient (or 0 if no value for this patient at all)
    #

    features_interpolated = interpolate_per_patient(train_features)
    train_features = drop_columns(features_interpolated, True)

    test_features_interpolated = interpolate_per_patient(test_features)
    test_features = drop_columns(test_features_interpolated, True)

    train_features.to_csv('data/train_features_interpolated.csv', index=False)
    test_features.to_csv('data/test_features_interpolated.csv', index=False)
    print("interpolated datasets have been generated.")
