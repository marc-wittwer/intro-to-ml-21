import pandas as pd
import datetime
import tensorflow as tf
import numpy as np

start = datetime.datetime.now()

labeled = pd.read_hdf('train_labeled.h5')
unlabeled = pd.read_hdf('train_unlabeled.h5')
test = pd.read_hdf('test.h5')
labels = labeled['y']
labeled_data = labeled.drop('y',axis=1)

inputx = labeled_data.iloc[:].values
inputy = labels.iloc[:].values
inputxp = unlabeled.iloc[:].values
outputx = test.iloc[:].values

inputx = tf.keras.utils.normalize(inputx)
inputxp = tf.keras.utils.normalize(inputxp)
outputx = tf.keras.utils.normalize(outputx)


layer_sizes = [80, 120, 160, 200, 240]
adam = 0.001
reg = 0.005
batch = 32
num_epoch = 500
winner_cond = 0.99

models = []
i = 0

for layer in layer_sizes:
	print("Training on labeled data with model layer size",layer)
	models.append(tf.keras.Sequential([
	tf.keras.layers.Dense(layer, activation="relu", bias_regularizer=tf.keras.regularizers.l2(reg)),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dense(layer, activation="relu", bias_regularizer=tf.keras.regularizers.l2(reg)),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dense(10, activation="softmax")]))

	models[i].compile(optimizer=tf.train.AdamOptimizer(adam), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	models[i].fit(inputx, inputy, validation_split=0.2, batch_size=batch, epochs=num_epoch)

	i = i + 1

wholex = inputx
wholey = inputy

inputxp1 = inputxp

for i in range(20):

	unlabeled_predictions = len(inputxp1)*[10*[0]]
	j = 0

	for model in models:
		unlabeled_predictions = unlabeled_predictions + models[j].predict(inputxp1)
		j = j + 1

	winnery = []
	winnerx = []
	inputxp1 = list(inputxp1)

	k = 0
	for pred in unlabeled_predictions:
		candidate = np.max(pred)
		if candidate/len(models) > winner_cond:
			winnerx.append(inputxp1[k])
			winnery.append(np.argmax(pred))
			inputxp1.pop(k)
			k = k - 1
		k = k + 1

	inputxp1 = np.asarray(inputxp1)
	winnery = np.asarray(winnery)
	winnerx = np.asarray(winnerx)

	if len(winnerx) > 0:
		wholex = np.concatenate((wholex,winnerx))
		wholey = np.concatenate((wholey,winnery))

	print("Added",len(winnery),"to training data")

	j = 0
	for model in models:
		print("Currently at iteration",i,"with model layer size",model.layers[0].units)
		models[j].fit(wholex, wholey, validation_split=0.2, batch_size=batch, epochs=num_epoch)
		j = j + 1

	output_predictions = len(outputx)*[10*[0]]

	for model in models:
		print("Last training loop in",i,"with model layer size",model.layers[0].units)
		output_predictions = output_predictions + model.predict(outputx)

	predictions_max = []
	for pred in output_predictions:
		predictions_max.append(np.argmax(pred))

	classifications = pd.DataFrame(predictions_max, columns=['y'])
	classifications.index.name = 'Id'
	classifications.index += 30000
	submission_name = 'submission5{:d}.csv'.format(i)
	classifications.to_csv(submission_name, header=True)



end = datetime.datetime.now()
exec_time = end-start
total = exec_time.total_seconds()
hours = (int) (total/60/60)
minutes = (int) ((total-hours*3600)/60)
seconds = (int) ((total-hours*3600-minutes*60)/60)
print("The whole execution took",hours,"h",minutes,"min",seconds,"s")