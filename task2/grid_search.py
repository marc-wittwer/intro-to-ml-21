import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

n_jobs = 3  # number of CPU cores to use

train_features = pd.read_csv('data/train_extracted_features.csv', index_col='pid')
train_features.sort_index(axis=0, ascending=True, inplace=True)
train_labels = pd.read_csv('data/train_labels.csv', index_col='pid')
train_labels.sort_index(axis=0, ascending=True, inplace=True)
test_features = pd.read_csv('data/test_extracted_features.csv')


# Subtask 1
medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2']

perform_grid_search = True
grid_search_params = {'svc__C': [0.1, 10, 100]}  # name is important

for test_label in medical_tests:
    pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler()),
                               ('svc', svm.SVC(class_weight='balanced'))])

    clf = GridSearchCV(pipeline, param_grid=grid_search_params, scoring='roc_auc', n_jobs=n_jobs, verbose=3, cv=None)  # cv=None means 5-fold cv
    clf.fit(train_features, train_labels[test_label])

    best_clf = clf.best_estimator_  # use this to make predictions on test set
    results_df = pd.DataFrame({'param': clf.cv_results_["params"], 'mean_score': clf.cv_results_["mean_test_score"],
                               'fit_time': clf.cv_results_["mean_fit_time"]})
    print(results_df)
