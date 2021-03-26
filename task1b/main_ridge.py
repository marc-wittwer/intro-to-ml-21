import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import math

#
# read data (as a pd.DataFrame)
#
train_data = pd.read_csv('data/train.csv')

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

#
# Fit model
#
alphas = np.arange(1.6, 1.8, 0.001)
model = linear_model.RidgeCV(alphas=alphas, cv=None,
                             fit_intercept=True, scoring=None,
                             normalize=True, store_cv_values=True)

X = features[features.columns]
Y = train_data['y']
model.fit(X, Y)
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)
cv_val = model.cv_values_
cv_mean = np.mean(cv_val, axis=0)
# plt.plot(alphas, cv_mean)
# plt.show()
print("Alpha: ", model.alpha_)

Y_predict = model.predict(X)
score = metrics.mean_squared_error(Y_predict, Y)**0.5
print("Training Score: ", score)

sol = pd.DataFrame(np.append(model.coef_[0:-1], model.intercept_))
sol.to_csv('data/solution_ridge.csv', index=False, header=False)
