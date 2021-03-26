import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#
# read data (as a pd.DataFrame)
#
train_data = pd.read_csv('data/train.csv')
train_data.plot()
plt.show()
alpha = [0.1, 1, 10, 100, 200]

#
# Fit model
#
xi_columns = train_data.columns[1:]
X = train_data[xi_columns]
print(X)

Y = train_data['y']
scores_alpha = []
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for a in alpha:
    model = linear_model.Ridge(alpha=a)
    scores = []

    for (train, test) in kfold.split(X, Y):
        model.fit(X.iloc[train, :], Y.iloc[train])
        Y_predict = model.predict(X.iloc[test, :])
        score = metrics.mean_squared_error(Y_predict, Y.iloc[test])**0.5
        scores.append(score)

    scores_alpha.append(np.mean(scores))

#
# Save solution
#
sol = pd.DataFrame(scores_alpha)
sol.to_csv('data/sample.csv', index=False, header=False)
