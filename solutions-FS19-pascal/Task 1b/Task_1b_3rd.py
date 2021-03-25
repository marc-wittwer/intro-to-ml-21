import csv
from sklearn import linear_model
import pandas
import math
import numpy
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

train = pandas.read_csv('train.csv')


# split data into y and X
X_train = train.drop(["Id", "y"], axis=1)
y_train = train[["y"]]

# transform variables and add to dataframe
# quadratic
X_train['phi6'] = X_train.x1*X_train.x1
X_train['phi7'] = X_train.x2*X_train.x2
X_train['phi8'] = X_train.x3*X_train.x3
X_train['phi9'] = X_train.x4*X_train.x4
X_train['phi10'] = X_train.x5*X_train.x5
# exponential
X_train['phi11'] = numpy.exp(X_train.x1)
X_train['phi12'] = numpy.exp(X_train.x2)
X_train['phi13'] = numpy.exp(X_train.x3)
X_train['phi14'] = numpy.exp(X_train.x4)
X_train['phi15'] = numpy.exp(X_train.x5)
# cosine
X_train['phi16'] = numpy.cos(X_train.x1)
X_train['phi17'] = numpy.cos(X_train.x2)
X_train['phi18'] = numpy.cos(X_train.x3)
X_train['phi19'] = numpy.cos(X_train.x4)
X_train['phi20'] = numpy.cos(X_train.x5)
# constant
X_train['phi21'] = 1


# run regression
reg3 = sm.OLS(y_train, X_train)
results3 = reg3.fit()

# predict y of training data and compute training error
train_predictions = results3.predict(X_train)
RMSE_training = mean_squared_error(y_train, train_predictions)**0.5
print(RMSE_training)

# write submission file
submission3 = pandas.DataFrame(data=results3.params)

submission3.to_csv("submission3.csv",index = False, header=False)

