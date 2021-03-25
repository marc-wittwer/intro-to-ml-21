# Import packages
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture

# Import data
train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

d = pd.DataFrame(train_unlabeled)
gmm = GaussianMixture(n_components=10, n_init=1000, tol=0.001)

# Fit the GMM model for the dataset
# which expresses the dataset as a
# mixture of 10 Gaussian Distribution
gmm.fit(d)

# Assign a label to each sample
y = gmm.predict(d)
d['y'] = y
d0 = d[d['y'] == 0]
d1 = d[d['y'] == 1]
d2 = d[d['y'] == 2]
d3 = d[d['y'] == 3]
d4 = d[d['y'] == 4]
d5 = d[d['y'] == 5]
d6 = d[d['y'] == 6]
d7 = d[d['y'] == 7]
d8 = d[d['y'] == 8]
d9 = d[d['y'] == 9]

train = train_labeled.append(d, ignore_index=True)

train.to_csv("train_all_labeled.csv", index=False, header=True)