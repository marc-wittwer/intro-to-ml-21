
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split


#
# read data (as a pd.DataFrame)
#

train_features = pd.read_csv('data/train_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')
test_features = pd.read_csv('data/test_features.csv')

# how many cores to use, adapt to availability
n_jobs = 7

#
# Replace nans with interpolation per patient (or 0 if no value for this patient at all)
#

def interpolate_patient(pid, features):
    pat_features = features.loc[features['pid'] == pid].interpolate(method='linear', axis=0, limit_direction='both')
    pat_features.fillna(0, inplace=True)

    features_flat = pat_features.set_index('pid').groupby(level=0).apply(
        lambda df: df.reset_index(drop=True)).unstack()
    features_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in features_flat.columns]
    return features_flat


def interpolate_per_patient(features):
    print("Start interpolate per patient:")

    # Parallelization
    pool = Pool(n_jobs)
    interpolated_features = pd.concat(pool.map(partial(interpolate_patient, features=features), features['pid'].unique()))

    print("Finished interpolation per patient")

    return interpolated_features


def drop_columns(features, drop_time_columns):
    # Drop Age/Time duplicates columns
    time_column_names = ['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10',
                         'Time11', 'Time12']
    age_column_names = ['Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Age7', 'Age8', 'Age9', 'Age10', 'Age11', 'Age12']
    columns_to_drop = time_column_names + age_column_names if drop_time_columns else age_column_names

    features_dropped = features.drop(columns_to_drop, axis=1)

    return features_dropped


features_interpolated = interpolate_per_patient(train_features)
train_features = drop_columns(features_interpolated, True)

#
# Split into train and test split
#
test_size = 0.9
train_X, test_X, train_y, test_y = train_test_split(train_features, train_labels, test_size=test_size, random_state=42)

def generate_model_report(y_actual, y_predicted):
    print("Accuracy = ", accuracy_score(y_actual, y_predicted))
    print("Precision = ", precision_score(y_actual, y_predicted))
    print("Recall = ", recall_score(y_actual, y_predicted))
    print("F1 Score = ", f1_score(y_actual, y_predicted))
    pass


def generate_auc_roc_curve(clsf, x_test, y_test):
    y_pred_proba = clsf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass

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
clf = svm.SVC(kernel='rbf', cache_size=1000, class_weight='balanced')
clf.fit(X, y)
print("\nSepsis prediction\n average accuracy: " + str(clf.score(test_X, y_test)) +
      "\n roc_auc score: " + str(roc_auc_score(y_test, np.sign(clf.decision_function(test_X)))))


# Task 3: Train regressor to predict mean value of vital signs
vital_signs = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
print()
print("\nstart regression for vital signs")
X = train_X
y = train_y[vital_signs]
y_test = test_y[vital_signs]
reg = MultiOutputRegressor(linear_model.Ridge(0.01), n_jobs=n_jobs)  # lots of parameters that could be set here
reg.fit(X, y)
print("\nVital signs prediction\n R2 score: " + str(r2_score(reg.predict(test_X), y_test)))


task1_score = np.mean(np.array(medical_test_scores)[:, 1])
task2_score = roc_auc_score(test_y['LABEL_Sepsis'], clf.decision_function(test_X))
task3_score = np.mean([0.5 + 0.5 * np.maximum(0, r2_score(reg.predict(test_X), test_y[vital_signs]))])

total_score = np.mean([task1_score, task2_score, task3_score])

print("\nScores:\n\nTask 1: " + str(task1_score) + "\nTask 2: " + str(task2_score) + "\nTask 3: " + str(
    task3_score) + "\n\nTotal Score: " + str(total_score))
print("\nBaseline: " + str(0.713853457215))
