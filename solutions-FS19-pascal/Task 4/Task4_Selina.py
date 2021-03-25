import pandas as pd
from sklearn.semi_supervised import LabelSpreading
import numpy as np


#Import data
train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")


#Put labeled and unlabled together
train = train_labeled.append(train_unlabeled, ignore_index=True)

#Set the NaN label values from the unlabeled train dataset to - 1
train = train.fillna(-1)

# Split the training data into y and X.
X_train = train.drop(["y"], axis=1)
y_train = train[["y"]]
print(X_train)
print(y_train)

#Noch parameter definieren!!!
label_prop_model = LabelSpreading()

label_prop_model.fit(X_train, np.ravel(y_train))

