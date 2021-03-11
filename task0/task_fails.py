import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt

#
# read data (as a pd.DataFrame)
#
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

#
# inspect data
#
train_data.info()  # counts, data types, quick overview
print(train_data.describe())  # statistics for numerical columns
train_data.plot()  # plot data using matplotlib
# train_data.plot(x='x1', y='y', kind='scatter')
plt.show()
# plt.savefig("train_data.png")  # if plt.show() doesnt work for you

#
# clean data?
#
# ensure no missing data, filter data (e.g. outliers),...

#
# Fit model
#
X = train_data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
Y = train_data['y']
model = linear_model.LinearRegression()
model.fit(X, Y)

#
# Test model
#
X_test = test_data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
Y_solution = X_test.mean(axis=1)  # we know y = mean(x_i)
Y_predictions = model.predict(X_test)
print(Y_predictions)
rms_error = metrics.mean_squared_error(Y_predictions, Y_solution) ** 0.5
print("RMS Error: ", rms_error)
aSDIOEQdh
#
# Save solution
#
sol = pd.concat([test_data['Id'], pd.DataFrame(Y_predictions)], axis=1)
sol.columns = ['Id', 'y']  # rename columns
# print(sol)
# sol.to_csv('data/sample_template.csv')
