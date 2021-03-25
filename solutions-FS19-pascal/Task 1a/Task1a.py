#Import required packages
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import make_scorer

#Import the dataset and define the variables
y = np.genfromtxt("train.csv", delimiter="," , skip_header=1, usecols=(1))
X = np.genfromtxt("train.csv", delimiter="," , skip_header=1, usecols=(2,3,4,5,6,7,8,9,10,11))


#Define the (negative) root mean squared error as a scoring function
def nrmse0(y_true, y_pred):
    return ((-1)*(((y_true - y_pred)**2).mean())**(1/2))
nrmse = make_scorer(nrmse0)


#Run the 10-fold cross-validation with the ridge regression for each alpha
ridge = Ridge()

parameters1 = {"alpha": [0.1]}
ridge_regressor1 = GridSearchCV(ridge, parameters1, scoring=nrmse, cv=10, iid=False)
ridge_regressor1.fit(X,y)
score1 =(-1)*ridge_regressor1.score(X, y)
print(score1)

parameters2 = {"alpha": [1]}
ridge_regressor2 = GridSearchCV(ridge, parameters2, scoring=nrmse, cv=10, iid=False)
ridge_regressor2.fit(X,y)
score2 =(-1)*ridge_regressor2.score(X, y)
print(score2)


parameters3 = {"alpha": [10]}
ridge_regressor3 = GridSearchCV(ridge, parameters3, scoring=nrmse, cv=10, iid=False)
ridge_regressor3.fit(X,y)
score3 =(-1)*ridge_regressor3.score(X, y)
print(score3)

parameters4 = {"alpha": [100]}
ridge_regressor4 = GridSearchCV(ridge, parameters4, scoring=nrmse, cv=10, iid=False)
ridge_regressor4.fit(X,y)
score4 =(-1)*ridge_regressor4.score(X, y)
print(score4)

parameters5 = {"alpha": [1000]}
ridge_regressor5 = GridSearchCV(ridge, parameters5, scoring=nrmse, cv=10, iid=False)
ridge_regressor5.fit(X,y)
score5 =(-1)*ridge_regressor5.score(X, y)
print(score5)

scores = np.array([score1, score2, score3, score4, score5])
print(scores)

#Save datafile
np.savetxt("finalfile.csv", scores, fmt='%1.8f')
