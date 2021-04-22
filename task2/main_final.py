import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Lasso
import time

n_jobs = 3  # number of CPU cores to use
np.random.seed(42)  # global random seed to get reproducible results

train_features = pd.read_csv('data/train_extracted_features.csv', index_col='pid')
train_labels = pd.read_csv('data/train_extracted_labels.csv', index_col='pid')
test_features = pd.read_csv('data/test_extracted_features.csv', index_col='pid')

train_features = train_features[0:12*10]
train_labels = train_labels[0:12*10]

test_predictions = []  # predictions on test set will be stored here


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


start_time = time.time()
t = start_time

# Subtask 1 + 2
medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2', 'LABEL_Sepsis']

lambdas_clf = [1, 0.1, 1, 0.1, 1, 0.1, 0.1, 1, 0.1, 1, 0.15]  # regularization parameters

for i, test_label in enumerate(medical_tests):
    print("\nFitting " + test_label + " ...")
    clf = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler()),
                          ('sel', SelectKBest(mutual_info_classif, k=100)),
                          ('svc', svm.SVC(class_weight='balanced', C=lambdas_clf[i], probability=True))])

    clf.fit(train_features, train_labels[test_label])
    test_predictions.append(clf.predict_proba(test_features)[:, 1])
    print("  " + str(i) + "/" + str(len(medical_tests)) + " completed")
    print("  Time elapsed (this test / total) ---> " + timer(t, time.time()) + " / " + timer(start_time, time.time()))
    t = time.time()


# Subtask 3
vital_signs = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
lambdas_reg = [1, 1.2, 0.1, 0.11]

for i, sign in enumerate(vital_signs):
    print("\nFitting " + sign + " ...")
    reg = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler()),
                          ('reg', Lasso(alpha=lambdas_reg[i]))])

    reg.fit(train_features, train_labels[sign])
    test_predictions.append(reg.predict(test_features))
    print("  Time elapsed (this test / total) [s]: " + timer(t, time.time()) + " / " + timer(start_time, time.time()))
    t = time.time()


# generate output
print("\nStoring result ...")
sol = pd.DataFrame(np.stack([test_predictions[i] for i in range(len(test_predictions))], axis=-1),
                   index=test_features.index)
sol.columns = medical_tests + vital_signs
sol.to_csv('data/prediction.csv', index=True, float_format='%.3f')
print("Done!")



