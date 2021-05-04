import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import f1_score

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_encoded_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')

train_features = train_features[:5000]
train_labels = train_labels[:5000]

test_features = pd.read_csv('data/test_encoded_features.csv')

n_jobs = 7

grid_search_params = [
    {'clf': svm.SVC(),
     'params': {'clf__C': [0.1, 1, 100],
                'clf__class_weight': ['balanced', None]}},

    ]  # name is important

results = []
best_estimators = []

for params in grid_search_params:
    pipeline = Pipeline(steps=[('clf', params['clf'])])

    grid_search = GridSearchCV(pipeline, param_grid=params['params'], scoring='f1', n_jobs=n_jobs, verbose=3, cv=None)

    grid_search.fit(train_features, train_labels["Active"])

    results.append(pd.DataFrame({'classifier': params['clf'], 'param': grid_search.cv_results_["params"],
                                'mean_score': grid_search.cv_results_["mean_test_score"],
                               'fit_time': grid_search.cv_results_["mean_fit_time"]}))

    best_estimators.append(grid_search.best_estimator_)

    print('Best parameters:', grid_search.best_params_, 'F1-Score:', grid_search.best_score_)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    for i, res in enumerate(results):
        print(res)

    # Predict labels for test data
    # best_clf = grid_search.best_estimator_
    # y_test_predictions = best_clf.predict(test_features)
#

# #  Save predictions to file
# y_test_predictions.to_csv('data/predictions.csv', index=False)
