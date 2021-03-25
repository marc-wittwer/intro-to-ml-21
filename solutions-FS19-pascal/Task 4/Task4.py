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
y_pred_unlabeled = pd.read_csv("y_pred_unlabeled.csv", header=None)

# combine labeled and unlabeled data
combined_X = [X_train_labeled_temp, X_train_unlabeled_temp]
combined_y = [y_train_labeled_temp, y_pred_unlabeled]

X_train_combined = pd.concat(combined_X)
y_train_combined = np.vstack(combined_y)

# Normalize the data by subtracting the mean and dividing through the standard deviation.
# Same mean and standard deviation (from train) used for test.
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train_combined)
X_test_scaled = sc_X.transform(X_test_temp)
y_train_scaled = np.ravel(y_train_combined)

# create model
model = Sequential()
model.add(Dense(100, input_dim=139, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

# compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# fit model
model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=50)

# evaluate  model
scores = model.evaluate(X_train_scaled, y_train_scaled)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred = model.predict_classes(X_test_scaled)

#Prepare submission file.
submission= pd.DataFrame(columns=['Id', 'y'])
i=0
for x in y_pred:
    submission.loc[i] = [str(i+30000),(y_pred[i])]
    i = i + 1
df = pd.DataFrame(["Id", "y"])
submission.append(df)

submission.to_csv("submission2.csv", index=False, header=True)