import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_encoded_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')

#train_features = train_features[:200]
#train_labels = train_labels[:200]

test_features = pd.read_csv('data/test_encoded_features.csv')

n_jobs = 3
np.random.seed(42)

grid_search_params = [
    # {'clf': svm.SVC(),
    #  'params': {'clf__C': [0.1, 1, 100, 1000],
    #             'clf__class_weight': ['balanced']}},

    {'clf': RandomForestClassifier(),
     'params': {'clf__n_estimators': [200],
                'clf__class_weight': ['balanced'],
                'clf__bootstrap': [True],
                'clf__min_samples_leaf': [1],
                'clf__min_samples_split': [5],
                'clf__max_depth': [None]
                }},

    {'clf': HistGradientBoostingClassifier(max_iter=1000, random_state=42),
     'params': {'clf__l2_regularization': [0.6, 0.7],
                'clf__max_leaf_nodes': [100, 150, 200],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__learning_rate': [0.1, 0.2],
                'clf__categorical_features': [[0, 1, 2, 3]]}}

    ]  # name is important

results = []
best_estimators = []

for params in grid_search_params:
    pipeline = Pipeline(steps=[('clf', params['clf'])])

    grid_search = GridSearchCV(pipeline, param_grid=params['params'], scoring='f1', n_jobs=n_jobs, verbose=3, cv=None)

    grid_search.fit(train_features, train_labels["Active"])

    # keep track of results
    df = pd.DataFrame({'classifier': str(params['clf']),
                       'mean_score': grid_search.cv_results_["mean_test_score"],
                       'fit_time': grid_search.cv_results_["mean_fit_time"]})
    for key in params["params"]:
        if len(params["params"][key]) > 1:
            df[key] = grid_search.cv_results_["param_" + key]
    results.append(df)

    # keep track of best estimator
    best_estimators.append((grid_search.best_estimator_, grid_search.best_score_))

    print('Best parameters:', grid_search.best_params_, 'F1-Score:', grid_search.best_score_)

# Display results for inspection
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    for i, res in enumerate(results):
        print("\n")
        print(res)
        print("Best estimator (with a score of " + str(best_estimators[i][1]) + "): ")
        print(best_estimators[i][0])

# Predict labels for test data with best classifier
best_estimators.sort(key=lambda x: x[1], reverse=True)  # sort by score (descending)
best_clf = best_estimators[0][0]
y_test_predictions = pd.DataFrame(best_clf.predict(test_features))

# #  Save predictions to file
y_test_predictions.to_csv('data/predictions.csv', index=False, header=False)
