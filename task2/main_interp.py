import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helpers import generate_model_report, generate_auc_roc_curve  # if that import doesnt work make sure to set "task2" as a source folder in "project structure" settings

use_features = True

if use_features:
    train_features = pd.read_csv('data/train_extracted_features.csv')
    train_labels = pd.read_csv('data/train_extracted_labels.csv')
    test_features = pd.read_csv('data/test_extracted_features.csv')
else:
    train_features = pd.read_csv('data/train_features_interpolated.csv')
    train_labels = pd.read_csv('data/train_labels_interpolated.csv')
    test_features = pd.read_csv('data/test_features_interpolated.csv')

train_features = train_features.set_index('pid')
train_labels = train_labels.set_index('pid')
test_features = test_features.set_index('pid')

train_idxs = train_features.index
test_idxs = test_features.index

imputeZero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
train_features = pd.DataFrame(imputeZero.fit_transform(train_features), columns=train_features.columns, index=train_idxs)
test_features = pd.DataFrame(imputeZero.fit_transform(test_features), columns=test_features.columns, index=test_idxs)

n_jobs = 7

#
# Split into train and test split
#


test_size = 0.9
train_X, test_X, train_y, test_y = train_test_split(train_features, train_labels, test_size=test_size, shuffle=False, random_state=42)

# Train classifiers to predict ordered medical tests
X = train_X
medical_test_scores = []
medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2']

for medical_test in medical_tests:
    print("\nstart SVM fitting for " + medical_test)
    y = train_y[medical_test]
    y_test = test_y[medical_test]

    print('Number of 1s and 0s in label data:\n', y.value_counts())

    clf = svm.SVC(kernel='rbf', cache_size=5000, class_weight='balanced')
    clf.probability = True
    clf.fit(X, y)
    # probabilities = clf.predict_proba(X)  # values between [0,1]

    y_test_predicted = clf.predict(test_X)
    # Log true/false positives
    print(pd.crosstab(pd.Series(y_test_predicted, name='Predicted'), pd.Series(y_test, name='Actual')))

    generate_model_report(y_test, y_test_predicted)
    # generate_auc_roc_curve(clf, test_X_flat, y_test)

    score = clf.score(test_X, y_test)
    auc_score = roc_auc_score(y_test, clf.decision_function(test_X))
    print('Roc auc score = ', auc_score)
    medical_test_scores.append([score, auc_score])

for i, test in enumerate(medical_tests):
    print("\n" + test + "\n average accuracy: " + str(medical_test_scores[i][0]) +
          "\n roc_auc score: " + str(medical_test_scores[i][1]))

task1_score = np.mean(np.array(medical_test_scores)[:, 1])
print("Mean auc score: ", task1_score)

# Task 2: Train classifier to predict sepsis event
print()
print("start svm fitting for sepsis prediction")
X = train_X
y = train_y['LABEL_Sepsis']
y_test = test_y['LABEL_Sepsis']
clf = svm.SVC(C=0.05, gamma='scale', kernel='rbf', cache_size=1000, probability=True, class_weight='balanced', random_state=42)
clf.fit(X, y)
print("\nSepsis prediction\n average accuracy: " + str(clf.score(test_X, y_test)) +
      "\n roc_auc score: " + str(roc_auc_score(y_test, clf.decision_function(test_X))))

# Task 3: Train regressor to predict mean value of vital signs
vital_signs = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
print()
print("\nstart regression for vital signs")
X = train_X
y = train_y[vital_signs]
y_test = test_y[vital_signs]
reg = MultiOutputRegressor(linear_model.RidgeCV(cv=None,
                             fit_intercept=True, scoring=None,
                             normalize=True, store_cv_values=True), n_jobs=n_jobs)  # lots of parameters that could be set here
reg.fit(X, y)
print("\nVital signs prediction\n R2 score: " + str(r2_score(reg.predict(test_X), y_test)))

# task1_score = np.mean(np.array(medical_test_scores)[:, 1])
task2_score = roc_auc_score(test_y['LABEL_Sepsis'], clf.decision_function(test_X))
task3_score = np.mean([0.5 + 0.5 * np.maximum(0, r2_score(reg.predict(test_X), test_y[vital_signs]))])

# total_score = np.mean([task1_score, task2_score, task3_score])
#
# print("\nScores:\n\nTask 1: " + str(task1_score) + "\nTask 2: " + str(task2_score) + "\nTask 3: " + str(
#     task3_score) + "\n\nTotal Score: " + str(total_score))
print("\nBaseline: " + str(0.713853457215))