import numpy as np
import pandas as pd
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns

if __name__ == "__main__":
    train_features = pd.read_csv('data/train_features_interpolated.csv')
    train_labels = pd.read_csv('data/train_labels.csv', index_col='pid')
    test_features = pd.read_csv('data/test_features_interpolated.csv')

    train_features = train_features[0:2 * 12]
    train_labels = train_labels[0:2 * 12]

    extracted_train_features = extract_features(train_features, column_id='pid', column_sort='Time1', n_jobs=3)
    print(extracted_train_features.shape)

    impute(extracted_train_features)  # to remove NaNs
    features_filtered = select_features(extracted_train_features, train_labels["LABEL_Sepsis"])
    kind_to_fc_parameters = from_columns(features_filtered.columns.tolist())
    print(kind_to_fc_parameters)
    print(features_filtered)
