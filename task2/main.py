import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
import math

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')
test_features = pd.read_csv('data/test_features.csv')

# Split train data into test data for internal scoring (0.7 means 70% training data/30% test data)
# Problem: The split happens sequentially and not randomly, eg. last rows are always testing data
train_test_split_ratio = 0.05

max_samples = train_labels.shape[0]
split_data_row_index = math.floor(train_test_split_ratio * max_samples)
print(split_data_row_index, 'samples will be used as training data.')

train_X = train_features[0:split_data_row_index * 12]
train_y = train_labels[0:split_data_row_index]

test_X = train_features[(split_data_row_index + 1) * 12:max_samples * 12]
test_y = train_labels[split_data_row_index + 1:max_samples]

medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2']

# Preprocessing of Data

# normalization
cols_to_norm = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb',
       'HCO3', 'BaseExcess', 'RRate', 'Fibrinogen', 'Phosphate', 'WBC',
       'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
       'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos',
       'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate',
       'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
for col in cols_to_norm:
    train_X = train_X.assign(col=preprocessing.StandardScaler().fit_transform(np.array(train_X[col]).reshape(-1, 1)))
    test_X = test_X.assign(col=preprocessing.StandardScaler().fit_transform(np.array(test_X[col]).reshape(-1, 1)))

cols_to_norm_y = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
for col in cols_to_norm_y:
    train_y = train_y.assign(col=preprocessing.StandardScaler().fit_transform(np.array(train_y[col]).reshape(-1, 1)))
    test_y = test_y.assign(col=preprocessing.StandardScaler().fit_transform(np.array(test_y[col]).reshape(-1, 1)))

# Missing Features / Imputation of NaN values

mean_std = True    # If to use mean and std per patient instead of flattening the timeseries (Might be good to extend by trend (slope of linear fit))
if mean_std:
    train_X_features = train_X.groupby(['pid']).mean()
    train_X_features = train_X_features.join(train_X.groupby(['pid']).std(), rsuffix='_std')
    train_X = train_X_features.reset_index()

    test_X_features = test_X.groupby(['pid']).mean()
    test_X_features = test_X_features.join(test_X.groupby(['pid']).std(), rsuffix='_std')
    test_X = test_X_features.reset_index()


# TODO: Idea: Use mean per patient
# TODO: Idea: Use mean per age group

# 'mean'          = mean of column
# 'zero'          = 0
# 'median'        = median of column
# 'most-frequent' = smallest most frequent value

selected_imputation_strategy = 'mean'

print("Start imputing NaN values")

if selected_imputation_strategy == 'mean':
    imputeMean = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_X_imputed = pd.DataFrame(imputeMean.fit_transform(train_X), columns=train_X.columns)
    test_X_imputed = pd.DataFrame(imputeMean.fit_transform(test_X), columns=test_X.columns)

elif selected_imputation_strategy == 'zero':
    imputeZero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    train_X_imputed = pd.DataFrame(imputeZero.fit_transform(train_X), columns=train_X.columns)
    test_X_imputed = pd.DataFrame(imputeZero.fit_transform(test_X), columns=test_X.columns)

elif selected_imputation_strategy == 'median':
    imputeMedian = SimpleImputer(missing_values=np.nan, strategy='median')
    train_X_imputed = pd.DataFrame(imputeMedian.fit_transform(train_X), columns=train_X.columns)
    test_X_imputed = pd.DataFrame(imputeMedian.fit_transform(test_X), columns=test_X.columns)

elif selected_imputation_strategy == 'most-frequent':
    imputeMostFrequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    train_X_imputed = pd.DataFrame(imputeMostFrequent.fit_transform(train_X), columns=train_X.columns)
    test_X_imputed = pd.DataFrame(imputeMostFrequent.fit_transform(test_X), columns=test_X.columns)
else:
    print('Imputation strategy is not selected.')

print("Finished imputing NaN values")

# Example using add_indicator = True for SimpleImputer
# imputeMean = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=add_indicator)
# train_X_imputed = pd.DataFrame(imputeMean.fit_transform(train_X), columns=train_X.columns.append('imputed_' + train_X.columns[train_X.isna().sum() > 0]))


# TODO: Filter out age/time columns


# Fix imbalanced classification data
# TODO: Add balancing logic


# Reshape patient data to single row
train_X_flat = train_X_imputed.set_index('pid').groupby(level=0).apply(
    lambda df: df.reset_index(drop=True)).unstack()
train_X_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in train_X_flat.columns]

test_X_flat = test_X_imputed.set_index('pid').groupby(level=0).apply(
    lambda df: df.reset_index(drop=True)).unstack()
test_X_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in test_X_flat.columns]

# set pid as index, not used  for prediction, since probably rather random
if train_X_flat.index.name != 'pid':
    train_X_flat.set_index('pid')
    test_X_flat.set_index('pid')

# Task 1: Medical tests
#  - SVC parameters to tune systematically
#       kernel functions, C value (regularization), decision_function_shape
#       class weight = can handle unbalanced classification data,
#       (random_state (seed))


# Train classifiers to predict ordered medical tests
X = train_X_flat
medical_test_scores = []

for medical_test in medical_tests:
    print("start SVM fitting for " + medical_test)
    y = train_y[medical_test]
    y_test = test_y[medical_test]
    clf = svm.SVC(kernel='rbf', cache_size=5000, class_weight='balanced')
    clf.probability = True
    clf.fit(X, y)
    # probabilities = clf.predict_proba(X)  # values between [0,1]

    score = clf.score(test_X_flat, y_test)
    auc_score = roc_auc_score(y_test, clf.decision_function(test_X_flat))
    medical_test_scores.append([score, auc_score])

for i, test in enumerate(medical_tests):
    print("\n" + test + "\n average accuracy: " + str(medical_test_scores[i][0]) +
          "\n roc_auc score: " + str(medical_test_scores[i][1]))

task1_score = np.mean(np.array(medical_test_scores)[:, 1])
print("Mean auc score: ", task1_score)

# Task 2: Train classifier to predict sepsis event
print("start svm fitting for sepsis prediction")
X = train_X_flat
y = train_y['LABEL_Sepsis']
y_test = test_y['LABEL_Sepsis']
clf = svm.SVC(kernel='rbf', cache_size=1000, class_weight='balanced')
clf.fit(X, y)
print("\nSepsis prediction\n average accuracy: " + str(clf.score(test_X_flat, y_test)) +
      "\n roc_auc score: " + str(roc_auc_score(y_test, np.sign(clf.decision_function(test_X_flat)))))

# Task 3: Train regressor to predict mean value of vital signs
print("\nstart linear fitting for vital signs")
X = train_X_flat
y = train_y[['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
y_test = test_y[['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
reg = MultiOutputRegressor(linear_model.LinearRegression())  # lots of parameters that could be set here
reg.fit(X, y)
print("\nVital signs prediction\n R2 score: " + str(r2_score(reg.predict(test_X_flat), y_test)))

# # Evaluate strategies
# results = list()
# strategies = ['mean', 'median', 'most_frequent', 'constant']
# for s in strategies:
#    # create the modeling pipeline
#    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())]) # Replace Classifier by model
#    # evaluate the model
#    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
#    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#    # store results
#    results.append(scores)
#    print('>%s %.3f (%.3f)' % (s, np.mean(scores), np.std(scores)))


# plot model performance for comparison
# plt.boxplot(results, labels=strategies, showmeans=True)
# plt.show()


task1_score = np.mean(np.array(medical_test_scores)[:, 1])
task2_score = roc_auc_score(test_y['LABEL_Sepsis'], clf.decision_function(test_X_flat))
task3_score = np.mean([0.5 + 0.5 * np.maximum(0, r2_score(reg.predict(test_X_flat), test_y[
    ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]))])

total_score = np.mean([task1_score, task2_score, task3_score])

print("\nScores:\n\nTask 1: " + str(task1_score) + "\nTask 2: " + str(task2_score) + "\nTask 3: " + str(
    task3_score) + "\n\nTotal Score: " + str(total_score))
print("\nBaseline: " + str(0.713853457215))

