import pandas as pd
from multiprocessing import Pool
from functools import partial

tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess', 'RRate', 'Fibrinogen',
         'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'ABPm',
         'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Chloride',
         'Hct', 'Heartrate', 'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
appendixes = ['_median', '_std', '_min', '_max', '_nTests']
columns_only_mean = ['Time', 'Age']


def extract_features(features):
    # Extract features per patient
    # mean
    features_flat = features.groupby(['pid']).mean()
    # median
    features_flat = features_flat.join(features.groupby(['pid']).median(), rsuffix=appendixes[0])
    # std
    features_flat = features_flat.join(features.groupby(['pid']).std(), rsuffix=appendixes[1])
    # min
    features_flat = features_flat.join(features.groupby(['pid']).min(), rsuffix=appendixes[2])
    # max
    features_flat = features_flat.join(features.groupby(['pid']).max(), rsuffix=appendixes[3])
    # num Tests
    features_flat = features_flat.join(features.notna().groupby(features['pid']).sum().drop('pid', axis=1), rsuffix=appendixes[4])

    return features_flat


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


def drop_columns(features, drop_time_columns, use_features):
    if use_features:
        features_dropped = features
        for appendix in appendixes:
            for col in columns_only_mean:
                features_dropped = features_dropped.drop(col+appendix, axis=1)
        if drop_time_columns:
            features_dropped = features_dropped.drop('Time', axis=1)
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
    train_labels = pd.read_csv('data/train_labels.csv', index_col='pid')
    test_features = pd.read_csv('data/test_features.csv')

    # how many cores to use, adapt to availability
    n_jobs = 7

    # use features (mean, std implemented) instead of concatenating
    use_features = True

    # If time is to be dropped
    drop_time = False

    if use_features:
        #
        # Extract features per patient
        #
        features_interpolated = extract_features(train_features)
        test_features_interpolated = extract_features(test_features)
    else:
        #
        # Replace nans with interpolation per patient (or 0 if no value for this patient at all)
        #
        features_interpolated = interpolate_per_patient(train_features)
        test_features_interpolated = interpolate_per_patient(test_features)

    train_features = drop_columns(features_interpolated, drop_time, use_features)
    test_features = drop_columns(test_features_interpolated, drop_time, use_features)

    train_features.sort_index(axis=0, ascending=True, inplace=True)
    train_labels.sort_index(axis=0, ascending=True, inplace=True)
    test_features.sort_index(axis=0, ascending=True, inplace=True)

    if use_features:
        train_features.to_csv('data/train_extracted_features.csv', index=True)
        train_labels.to_csv('data/train_extracted_labels.csv', index=True)
        test_features.to_csv('data/test_extracted_features.csv', index=True)
    else:
        train_features.to_csv('data/train_features_interpolated.csv', index=True)
        train_labels.to_csv('data/train_labels_interpolated.csv', index=True)
        test_features.to_csv('data/test_features_interpolated.csv', index=True)
    print("interpolated datasets have been generated.")
