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
# train_data[train_data.columns[2:]].plot()
# plt.show()


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
features = pd.concat([lin_features, quad_features, exp_features, cos_features], axis=1)

#
# Fit model
#
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model = linear_model.LassoCV(alphas=None, n_alphas=1000, cv=350,
                             fit_intercept=True, verbose=True,
                             normalize=True, random_state=42,
                             max_iter=100000, n_jobs=7, selection='random')

X = features[features.columns]
Y = train_data['y']
model.fit(X, Y)
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)
cv_val = model.mse_path_
cv_mean = np.mean(cv_val, axis=1)
print("Picked alpha: ", model.alpha_)
print("Cross-validation mean: ", cv_mean[-1])
plt.plot(model.alphas_, cv_mean)
plt.show()

Y_predict = model.predict(X)
score = metrics.mean_squared_error(Y_predict, Y)**0.5
print("Score: ", score)

sol = pd.DataFrame(np.append(model.coef_, model.intercept_))
sol.to_csv('data/solution_lasso.csv', index=False, header=False)
