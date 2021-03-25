#Import required packages.
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# set random seed
np.random.seed(26)



# Import the data.
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
train_labeled = pd.read_hdf("train_labeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

# add column of -1 for missing labels
train_unlabeled.insert(loc=0, column='y', value=-1)

# Split the training data into y and X.
X_train_labeled_temp = train_labeled.drop(["y"], axis=1)
y_train_labeled_temp = train_labeled[["y"]]
X_train_unlabeled_temp = train_unlabeled.drop(["y"], axis=1)
y_train_unlabeled_temp = train_unlabeled[["y"]]
# Split test data into X.
X_test_temp = test


# combine labeled and unlabeled data
combined_X = [X_train_labeled_temp, X_train_unlabeled_temp]
combined_y = [y_train_labeled_temp, y_train_unlabeled_temp]

X_train_temp = pd.concat(combined_X)
y_train_temp = pd.concat(combined_y)

# create model
model = Sequential()
model.add(Dense(100, input_dim=139, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="relu"))

# compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# fit model
model.fit(X_train_temp, y_train_temp, epochs=50, batch_size=50)

# evaluate  model
scores = model.evaluate(X_train_temp, y_train_temp)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


