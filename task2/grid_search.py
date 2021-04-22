import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

n_jobs = 3  # number of CPU cores to use
np.random.seed(42)  # global random seed to get reproducible results

train_features = pd.read_csv('data/train_extracted_features.csv', index_col='pid')
train_labels = pd.read_csv('data/train_extracted_labels.csv', index_col='pid')
test_features = pd.read_csv('data/test_extracted_features.csv')


# Subtask 1
#medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
#                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
#                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
#                 'LABEL_EtCO2']

# Subtask 2
medical_tests = [] #['LABEL_Sepsis']

grid_search_params = {'svc__C': [0.1, 0.15, 0.2],
                      'imputer__strategy': ['mean', 'constant']}  # name is important

for test_label in medical_tests:
    pipeline = Pipeline(steps=[('imputer', SimpleImputer(fill_value=0)),
                               ('scaler', StandardScaler()),
                               ('sel', SelectKBest(mutual_info_classif, k=100)),
                               ('svc', svm.SVC(class_weight='balanced'))])

    clf = GridSearchCV(pipeline, param_grid=grid_search_params, scoring='roc_auc', n_jobs=n_jobs, verbose=3, cv=None)
    clf.fit(train_features, train_labels[test_label])

    best_clf = clf.best_estimator_  # use this to make predictions on test set
    results_df = pd.DataFrame({'param': clf.cv_results_["params"], 'mean_score': clf.cv_results_["mean_test_score"],
                               'fit_time': clf.cv_results_["mean_fit_time"]})
    print(results_df)


# Subtask 3
vital_signs = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
grid_search_params_3 = {'reg__alpha': [0.1, 1, 10, 100, 1000, 10000]}
results = []
for sign in vital_signs:
    print("fitting " + sign)
    pipeline_3 = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                 ('scaler', StandardScaler()),
                                 #('sel', SelectKBest(f_regression, k=100)),
                                 ('reg', Lasso())])

    model = GridSearchCV(pipeline_3, param_grid=grid_search_params_3, scoring='r2', n_jobs=n_jobs, verbose=3, cv=None)
    model.fit(train_features, train_labels[sign])
    results.append(pd.DataFrame({'param': model.cv_results_["params"],
                                 'mean_score': model.cv_results_["mean_test_score"],
                                 'fit_time': model.cv_results_["mean_fit_time"]}))

for i, result in enumerate(results):
    print("\n --- " + vital_signs[i] + " ---")
    print(result)
