import csv
from sklearn import linear_model
import pandas
import numpy
from sklearn.metrics import mean_squared_error
train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

# split data into y and X
X_train = train.drop(["Id", "y"], axis=1)
y_train = train[["y"]]
X_test = test.drop(["Id"], axis=1)

# run regressions
reg1 = linear_model.LinearRegression()
reg1.fit(X_train, y_train)

# predict y of training data
train_predictions = reg1.predict(X_train)

# compute training error
RMSE_training = mean_squared_error(y_train, train_predictions)**0.5
print(RMSE_training)

# predict y for test set
test_predictions = reg1.predict(X_test)

# create submission table
submission = pandas.DataFrame(columns=["Id", "y"])
i=0
for y in test_predictions:
    submission.loc[i] = [str(10000+i), (test_predictions[i])[0]]
    i=i+1



# write submission file
submission.to_csv("submission.csv", sep=",", index = False)










