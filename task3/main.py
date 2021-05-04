import pandas as pd

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_encoded_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')

test_features = pd.read_csv('data/test_encoded_features.csv')

