#Import required packages.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import datetime

start = datetime.datetime.now()

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

layer_sizes = [150, 200, 250, 350]

models = []
i = 0

for size in layer_sizes:
	print("Training w/ labeled, current layer size: ",size)
	models.append(Sequential([
	Dense(size, activation="relu"),
	Dense(size, activati

































		on="relu"),
	Dense(10, activation="softmax")]))

	models[i].compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

	models[i].fit(X_train_labeled_scaled, y_train_labeled_scaled, validation_split=0.2, epochs=200, batch_size=50)

	i = i + 1

threshold = .995
X_train_iteration = X_train_unlabeled_scaled
X_combined = X_train_labeled_scaled
y_combined = y_train_labeled_scaled

for i in range(25):
	y_pred_unlabeled = len(X_train_iteration)*[10*[0]]
	j = 0

	for model in models:
		y_pred_unlabeled = y_pred_unlabeled + models[j].predict(X_train_iteration)
		j = j + 1

	y_selected = []
	X_selected = []
	X_train_iteration = list(X_train_iteration)

	k = 0
	for pred in y_pred_unlabeled:
		pot_mover = np.max(pred)
		if pot_mover/len(models) > threshold:
			X_selected.append(X_train_iteration[k])
			y_selected.append(np.argmax(pred))
			X_train_iteration.pop(k)
			k = k - 1
		k = k + 1

	X_train_iteration = np.asarray(X_train_iteration)
	y_selected = np.asarray(y_selected)
	X_selected = np.asarray(X_selected)

	if len(X_selected) > 0:
		X_combined = np.concatenate((X_combined,X_selected))
		y_combined = np.concatenate((y_combined,y_selected))

	print("Moved",len(y_selected),"to training data")

	j = 0
	for model in models:
		print("Current iteration:",i,"; layer size: ",model.layers[0].units)
		models[j].fit(X_combined, y_combined, validation_split=0.2, epochs=200, batch_size=50)
		j = j + 1

	output_predictions = len(X_test_scaled)*[10*[0]]

	for model in models:
		print("Final iteration: ",i,"; layer size: ",model.layers[0].units)
		output_predictions = output_predictions + model.predict(X_test_scaled)

	predictions_max = []
	for pred in output_predictions:
		predictions_max.append(np.argmax(pred))

	classifications = pd.DataFrame(predictions_max, columns=['y'])
	classifications.index.name = 'Id'
	classifications.index += 30000
	submission = 'submission_iteration{:d}_4th.csv'.format(i)
	classifications.to_csv(submission, header=True)


end = datetime.datetime.now()
exec_time = end-start
total = exec_time.total_seconds()
hours = (int) (total/3600)
minutes = (int) ((total-hours*3600)/60)
seconds = (int) ((total-hours*3600-minutes*60)/60)
print("Total execution time: ",hours,"h",minutes,"min",seconds,"s")