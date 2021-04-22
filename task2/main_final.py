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
test_features = pd.read_csv('data/test_extracted_features.csv', index_col='pid')

train_features = train_features[0:12*3]
train_labels = train_labels[0:12*3]

test_predictions = []  # predictions on test set will be stored here

# Subtask 1 + 2
medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2', 'LABEL_Sepsis']

lambdas_clf = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.15]  # regularization parameters

for i, test_label in enumerate(medical_tests):
    print("Fitting " + test_label + " ...")
    clf = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler()),
                          ('sel', SelectKBest(mutual_info_classif, k=100)),
                          ('svc', svm.SVC(class_weight='balanced', C=lambdas_clf[i], probability=True))])

    clf.fit(train_features, train_labels[test_label])
    test_predictions.append(clf.predict_proba(test_features)[:, 1])


# Subtask 3
vital_signs = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
lambdas_reg = [1, 1, 1, 1]

for i, sign in enumerate(vital_signs):
    print("Fitting " + sign + " ...")
    reg = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler()),
                          ('reg', Lasso(alpha=lambdas_reg[i]))])

    reg.fit(train_features, train_labels[sign])
    test_predictions.append(reg.predict(test_features))


# generate output
print("Storing result ...")
sol = pd.DataFrame(np.stack([test_predictions[i] for i in range(len(test_predictions))], axis=-1))
#test_pid = pd.DataFrame(np.array([test_features['pid'].iloc[i] for i in range(len(test_features)) if i % 12 == 0]))
#sol = pd.concat([test_pid, sol], axis=1)
#sol.columns = ['pid'] + medical_tests + ['LABEL_Sepsis'] + vital_signs
sol.columns = medical_tests + vital_signs
sol.to_csv('data/prediction.csv', index=True, float_format='%.3f')
print("Done!")



