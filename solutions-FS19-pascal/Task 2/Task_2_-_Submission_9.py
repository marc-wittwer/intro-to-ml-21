#Import required packages.
import pandas
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np

# Import the data.
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

# Split the training data into y and X.
X_train_temp = train.drop(["Id", "y"], axis=1)
y_train_temp = train[["y"]]
# Split test data into X.
X_test_temp = test.drop(["Id"], axis = 1)

# set seed
np.random.seed(26)

# Normalize the data by subtracting the mean and dividing through the standard error.
# Same mean and standard error (from train) used for test.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train_temp)
X_test = sc_X.transform(X_test_temp)
y_train = np.ravel(y_train_temp)


#Train and fit the model using neural networks.
clf = MLPClassifier(hidden_layer_sizes=(100,100), activation="relu", solver="adam", alpha=0.001, max_iter=10000)
clf.fit(X_train, y_train)

#test accuracy of predictions on training set
train_pred = clf.predict(X_train)
print(accuracy_score(y_train, train_pred))

# predict test data
y_pred =clf.predict(X_test)

#Prepare submission file.
submission= pandas.DataFrame(columns=['Id', 'y'])
i=0
for x in y_pred:
    submission.loc[i] = [str(i+2000),(y_pred[i])]
    i = i + 1
df = pandas.DataFrame(["Id", "y"])
submission.append(df)

submission.to_csv("submission9.csv", index=False, header=True)




