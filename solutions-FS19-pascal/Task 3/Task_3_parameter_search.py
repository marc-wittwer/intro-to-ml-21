#Import required packages.
import pandas
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

# Import the data.
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")


# Split the training data into y and X.
X_train_temp = train.drop(["y"], axis=1)
y_train_temp = train[["y"]]
# Split test data into X.
X_test_temp = test


# Normalize the data by subtracting the mean and dividing through the standard deviation.
# Same mean and standard deviation (from train) used for test.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train_temp)
X_test = sc_X.transform(X_test_temp)
y_train = np.ravel(y_train_temp)



#Train and fit the model using neural networks.
np.random.seed(26)
clf = MLPClassifier(max_iter=10000)

#Select from best model
parameter_space = {
    'hidden_layer_sizes': [(80, 80, 80), (50,100,50), (100,), (80, 80)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],

}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(clf, parameter_space, n_jobs=-5, cv=5)
clf.fit(X_train, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_pred =clf.predict(X_test)

#Prepare submission file.
submissionfinal3= pandas.DataFrame(columns=['Id', 'y'])
i=0
for x in y_pred:
    submissionfinal3.loc[i] = [str(i+45324),(y_pred[i])]
    i = i + 1
df = pandas.DataFrame(["Id", "y"])
submissionfinal3.append(df)

submissionfinal3.to_csv("submissionfinal3.csv", index=False, header=True)

