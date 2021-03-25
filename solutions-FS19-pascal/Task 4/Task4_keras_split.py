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

# Normalize the data by subtracting the mean and dividing through the standard deviation.
# Same mean and standard deviation (from train) used for test.
sc_X = StandardScaler()
X_train_labeled_scaled = sc_X.fit_transform(X_train_labeled_temp)
X_train_unlabeled_scaled = sc_X.transform(X_train_unlabeled_temp)
X_test_scaled = sc_X.transform(X_test_temp)
y_train_labeled_scaled = np.ravel(y_train_labeled_temp)


# create model
model = Sequential()
model.add(Dense(100, input_dim=139, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

# compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# fit model
model.fit(X_train_labeled_scaled, y_train_labeled_scaled, epochs=50, batch_size=50)

# evaluate  model
scores = model.evaluate(X_train_labeled_scaled, y_train_labeled_scaled)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred_unlabeled = model.predict_classes(X_train_unlabeled_scaled)

# np.savetxt("y_pred_unlabeled.csv", y_pred_unlabeled, delimiter=",")


# combine labeled and unlabeled data
# doesn't work somehow
combined_X = [X_train_labeled_temp, X_train_unlabeled_temp]
combined_y = [y_train_labeled_temp, y_pred_unlabeled]

X_train_combined = pd.concat(combined_X)
y_train_combined = pd.concat(combined_y)

