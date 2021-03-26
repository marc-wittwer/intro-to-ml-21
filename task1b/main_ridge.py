import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math

#
# read data (as a pd.DataFrame)
#
train_data = pd.read_csv('data/train.csv')
train_data[train_data.columns[2:]].plot()
plt.show()


x = train_data[train_data.columns[2:]]
lin_features = x
lin_features = lin_features.rename(columns=lambda name: str(name) + "_lin")
quad_features = x**2
quad_features = quad_features.rename(columns=lambda name: str(name) + "_quad")
exp_features = x.applymap(lambda xi: math.exp(xi))
exp_features = exp_features.rename(columns=lambda name: str(name) + "_exp")
cos_features = x.applymap(math.cos)
cos_features = cos_features.rename(columns=lambda name: str(name) + "_cos")
const_features = pd.DataFrame(1, index=np.arange(len(x)), columns=['const'])
features = pd.concat([lin_features, quad_features, exp_features, cos_features, const_features], axis=1)
print(features)

#
# Fit model
#
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
model = linear_model.RidgeCV(alphas=alphas, cv=None,
                             fit_intercept=False, scoring=None,
                             normalize=True, store_cv_values=True)

X = features[features.columns]
Y = train_data['y']
model.fit(X, Y)
weights = model.get_params()
print(weights)
print(model.coef_)
print(model.intercept_)
cv_val = model.cv_values_
cv_mean = np.mean(cv_val, axis=0)
plt.plot(alphas, cv_mean)
plt.show()

Y_predict = model.predict(X)
score = metrics.mean_squared_error(Y_predict, Y)**0.5
print(score)

sol = pd.DataFrame(model.coef_)
sol.to_csv('data/solution.csv', index=False, header=False)
