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

# Normalize the data by subtracting the mean and dividing through the standard error.
# Same mean and standard error (from train) used for test.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train_temp)
X_test = sc_X.transform(X_test_temp)
y_train = np.ravel(y_train_temp)


#Train and fit the model using neural networks.
clf = MLPClassifier(hidden_layer_sizes=(10000,10), activation="tanh", solver="adam", alpha=0.0001, max_iter=1000)
clf.fit(X_train, y_train)
y_pred =clf.predict(X_test)

#test accuracy of predictions on training set
train_pred = clf.predict(X_train)
print(accuracy_score(y_train, train_pred))

#Prepare submission file.
submission3= pandas.DataFrame(columns=['Id', 'y'])
i=0
for x in y_pred:
    submission3.loc[i] = [str(i+2000),(y_pred[i])]
    i = i + 1
df = pandas.DataFrame(["Id", "y"])
submission3.append(df)

submission3.to_csv("submission3.csv", index=False, header=True)




