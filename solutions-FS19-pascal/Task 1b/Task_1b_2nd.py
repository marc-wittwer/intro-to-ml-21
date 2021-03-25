import csv
from sklearn import linear_model
import pandas
import math
import numpy
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

train = pandas.read_csv('train.csv')


# split data into y and X
train = train.drop(["Id"], axis=1)


# transform variables and add to dataframe
# quadratic
train['phi6'] = train.x1*train.x1
train['phi7'] = train.x2*train.x2
train['phi8'] = train.x3*train.x3
train['phi9'] = train.x4*train.x4
train['phi10'] = train.x5*train.x5
# exponential
train['phi11'] = numpy.exp(train.x1)
train['phi12'] = numpy.exp(train.x2)
train['phi13'] = numpy.exp(train.x3)
train['phi14'] = numpy.exp(train.x4)
train['phi15'] = numpy.exp(train.x5)
# cosine
train['phi16'] = numpy.cos(train.x1)
train['phi17'] = numpy.cos(train.x2)
train['phi18'] = numpy.cos(train.x3)
train['phi19'] = numpy.cos(train.x4)
train['phi20'] = numpy.cos(train.x5)
# constant
train['phi21'] = 1


# run regression
reg2 = sm.OLS(formula="y ~ x1 + x2 + x3 + x4 + x5 + phi6 + phi7 + phi8 + phi9 + phi10 + phi11 + phi12 + phi13 + phi14 + phi15 + phi16 + phi17 + phi18 + phi19 + phi20 + phi21", data=train).fit()
print(reg2.params)
# # run regressions
# reg2 = sm.OLS(_train, train)
# print(reg2)
# print(reg2.params)
#
# # predict y of training data and compute training error
# train_predictions = reg2.predict(train)
# RMSE_training = mean_squared_error(y_train, train_predictions)**0.5
# print(RMSE_training)
#
# # write submission file
# submission = pandas.DataFrame(data=reg2.coef_)
# submissionT = submission.transpose()
# submissionT.to_csv("submission.csv",index = False, header=False)

