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

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')
test_features = pd.read_csv('data/test_features.csv')

train_begin = 0
train_end = 999
test_begin = 1000
test_end = 1999

train_X = train_features[train_begin*12:train_end*12]
train_y = train_labels[train_begin:train_end]

test_X = train_features[test_begin*12:test_end*12]
test_y = train_labels[test_begin:test_end]

medical_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2']

# Missing Features
# ------ indicator flags (dummy values) für imputed values
add_indicator = False

# ------ mean über ganze spalte
# imputeMean = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=add_indicator)
# train_imputeMean = pd.DataFrame(imputeMean.fit_transform(train_features), columns=train_features.columns)
# ------ 0 setzen
# imputeZero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0, add_indicator=add_indicator)
# train_imputeZero = pd.DataFrame(imputeZero.fit_transform(train_features), columns=train_features.columns)

# ------ median über spalte
# imputeMedian = SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=add_indicator)
# train_imputeMedian = pd.DataFrame(imputeMedian.fit_transform(train_features), columns=train_features.columns)

# (- häufigster wert)
# imputeMostFrequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent', add_indicator=add_indicator)
# train_imputeMostFreq = pd.DataFrame(imputeMostFrequent.fit_transform(train_features), columns=train_features.columns)



# Imbalanced Classification





print("Start imputing stuff")

imputeMean = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=add_indicator)
if add_indicator:
    train_X_imputed = pd.DataFrame(imputeMean.fit_transform(train_X), columns=train_X.columns.append('imputed_' + train_X.columns[train_X.isna().sum() > 0]))
    test_X_imputed = pd.DataFrame(imputeMean.fit_transform(test_X), columns=test_X.columns.append('imputed_' + test_X.columns[test_X.isna().sum() > 0]))
else:
    train_X_imputed = pd.DataFrame(imputeMean.fit_transform(train_X), columns=train_X.columns)
    test_X_imputed = pd.DataFrame(imputeMean.fit_transform(test_X), columns=test_X.columns)

# TODO: Filter out age/time columns

train_X_flat = train_X_imputed.set_index('pid').groupby(level=0) \
    .apply(lambda df: df.reset_index(drop=True)) \
    .unstack().sort_index(axis=1, level=1)
train_X_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in train_X_flat.columns]

test_X_flat = test_X_imputed.set_index('pid').groupby(level=0) \
    .apply(lambda df: df.reset_index(drop=True)) \
    .unstack().sort_index(axis=1, level=1)
test_X_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in test_X_flat.columns]

# train_features.groupby(['pid']).mean()
# pid AAA BBB CCC DDD                 pid AAA BBB CCC DDD AAA BBB CCC DDD AAA BBB CCC DDD
# 1   0   0   0   0                   1   0   0   0   0   0   0   0   0   0   0   0   0
# 1   0   0   0   0     reshape to
# 1   0   0   0   0

print("done!")

# Train classifiers to predict ordered medical tests
X = train_X_flat
medical_test_scores = []

for medical_test in medical_tests:
    print("start SVM fitting for " + medical_test)
    y = train_y[medical_test]
    y_test = test_y[medical_test]
    clf = svm.SVC(kernel='rbf', cache_size=1000)
    clf.probability = True
    clf.fit(X, y)
    #probabilities = clf.predict_proba(X)  # values between [0,1]
    medical_test_scores.append([clf.score(test_X_flat, y_test),
                                roc_auc_score(y_test, clf.decision_function(test_X_flat))])

for i, test in enumerate(medical_tests):
    print("\n" + test + "\n average accuracy: " + str(medical_test_scores[i][0]) +
          "\n roc_auc score: " + str(medical_test_scores[i][1]))


# Train classifier to predict sepsis event
print("start svm fitting for sepsis prediction")
X = train_X_flat
y = train_y['LABEL_Sepsis']
y_test = test_y['LABEL_Sepsis']
clf = svm.SVC(kernel='linear', cache_size=1000)
clf.fit(X, y)
print("\nSepsis prediction\n average accuracy: " + str(clf.score(test_X_flat, y_test)) +
      "\n roc_auc score: " + str(roc_auc_score(y_test, clf.decision_function(test_X_flat))))

# Train regressor to predict mean value of vital signs
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
#for s in strategies:
#    # create the modeling pipeline
#    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())]) # Replace Classifier by model
#    # evaluate the model
#    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
#    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#    # store results
#    results.append(scores)
#    print('>%s %.3f (%.3f)' % (s, np.mean(scores), np.std(scores)))


# plot model performance for comparison
#plt.boxplot(results, labels=strategies, showmeans=True)
#plt.show()

task1_score = np.mean(np.array(medical_test_scores)[:, 1])
task2_score = roc_auc_score(test_y['LABEL_Sepsis'], clf.decision_function(test_X_flat))
task3_score = np.mean([0.5 + 0.5 * np.maximum(0, r2_score(reg.predict(test_X_flat), test_y[['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]))])

total_score = np.mean([task1_score, task2_score, task3_score])

print("\nScores:\n\nTask 1: " + str(task1_score) + "\nTask 2: " + str(task2_score) + "\nTask 3: " + str(task3_score) + "\n\nTotal Score: " + str(total_score))
print("\nBaseline: " + str(0.713853457215))
