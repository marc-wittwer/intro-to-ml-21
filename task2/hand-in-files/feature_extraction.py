import pandas as pd

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


def drop_columns(features):
    features_dropped = features
    for appendix in appendixes:
        for col in columns_only_mean:
            features_dropped = features_dropped.drop(col+appendix, axis=1)

    return features_dropped


if __name__ == "__main__":

    #
    # read data (as a pd.DataFrame)
    #
    train_features = pd.read_csv('data/train_features.csv')
    train_labels = pd.read_csv('data/train_labels.csv', index_col='pid')
    test_features = pd.read_csv('data/test_features.csv')

    #
    # Extract features per patient (train and test)
    #
    features_interpolated = extract_features(train_features)
    test_features_interpolated = extract_features(test_features)

    # Drop columns for Time and age to retain only the mean (Regular data, others can easily be reconstructed from it)
    train_features = drop_columns(features_interpolated)
    test_features = drop_columns(test_features_interpolated)

    # sort data to have all in the same order
    train_features.sort_index(axis=0, ascending=True, inplace=True)
    train_labels.sort_index(axis=0, ascending=True, inplace=True)
    test_features.sort_index(axis=0, ascending=True, inplace=True)

    # write data
    train_features.to_csv('data/train_extracted_features.csv', index=True)
    train_labels.to_csv('data/train_extracted_labels.csv', index=True)
    test_features.to_csv('data/test_extracted_features.csv', index=True)

    print("interpolated datasets have been generated.")
