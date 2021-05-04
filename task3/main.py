import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_encoded_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')

# train_features = train_features[:20000]
# train_labels = train_labels[:20000]

test_features = pd.read_csv('data/test_encoded_features.csv')

n_jobs = 7
np.random.seed(42)

grid_search_params = [
    # {'clf': svm.SVC(),
    #  'params': {'clf__C': [0.1, 1, 100, 1000],
    #             'clf__class_weight': ['balanced']}},
    {'clf': HistGradientBoostingClassifier(max_iter=1000, random_state=42),
     'params': {'clf__l2_regularization': [0.4, 0.5, 0.6],
                'clf__max_leaf_nodes': [50, 100, 150],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__categorical_features': [[0, 1, 2, 3]]}}

    ]  # name is important

results = []
best_estimators = []

for params in grid_search_params:
    pipeline = Pipeline(steps=[('clf', params['clf'])])

    grid_search = GridSearchCV(pipeline, param_grid=params['params'], scoring='f1', n_jobs=n_jobs, verbose=3, cv=None)

    grid_search.fit(train_features, train_labels["Active"])

    results.append(pd.DataFrame({'classifier': str(params['clf']), 'param': grid_search.cv_results_["params"],
                                'mean_score': grid_search.cv_results_["mean_test_score"],
                               'fit_time': grid_search.cv_results_["mean_fit_time"]}))

    best_estimators.append(grid_search.best_estimator_)

    print('Best parameters:', grid_search.best_params_, 'F1-Score:', grid_search.best_score_)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    for i, res in enumerate(results):
        print(res)

# Predict labels for test data with best classifier (only one classifier -> only one best_estimator)
best_clf = best_estimators[0]
y_test_predictions = pd.DataFrame(best_clf.predict(test_features))

# #  Save predictions to file
y_test_predictions.to_csv('data/predictions.csv', index=False, header=False)
