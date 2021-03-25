#Import required packages.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# set random seed
np.random.seed(26)

# Import the data.
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
train_labeled = pd.read_hdf("train_labeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

# Split the training data into y and X.
X_train_labeled_temp = train_labeled.drop(["y"], axis=1)
y_train_labeled_temp = train_labeled[["y"]]
X_train_unlabeled_temp = train_unlabeled
# Split test data into X.
X_test_temp = test

# import previsouly predicted y's
y_pred_unlabeled = pd.read_csv("y_pred_unlabeled.csv")

# combine labeled and unlabeled data
combined_X = [X_train_labeled_temp, X_train_unlabeled_temp]
combined_y = [y_train_labeled_temp, y_pred_unlabeled]

X_train_combined = pd.concat(combined_X)
y_train_combined = np.vstack(combined_y)

print(y_train_combined)