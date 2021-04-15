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
from sklearn.metrics import roc_auc_score

#
# read data (as a pd.DataFrame)
#
train_features = pd.read_csv('data/train_features.csv')
train_labels = pd.read_csv('data/train_labels.csv')
test_features = pd.read_csv('data/test_features.csv')

train_features = train_features.head(100*12)
train_labels = train_labels.head(100)

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
train_imputed = pd.DataFrame(imputeMean.fit_transform(train_features), columns=train_features.columns)

# TODO: Filter out age/time columns

train_features_flat = train_imputed.set_index('pid').groupby(level=0) \
    .apply(lambda df: df.reset_index(drop=True)) \
    .unstack().sort_index(axis=1, level=1)
train_features_flat.columns = ['{}{}'.format(x[0], int(x[1]) + 1) for x in train_features_flat.columns]

# train_features.groupby(['pid']).mean()
# pid AAA BBB CCC DDD                 pid AAA BBB CCC DDD AAA BBB CCC DDD AAA BBB CCC DDD
# 1   0   0   0   0                   1   0   0   0   0   0   0   0   0   0   0   0   0
# 1   0   0   0   0     reshape to
# 1   0   0   0   0

print("done!")

# Train classifiers to predict ordered medical tests
X = train_features_flat
medical_test_scores = []

for medical_test in medical_tests:
    print("start SVM fitting for " + medical_test)
    y = train_labels[medical_test]
    clf = svm.SVC(kernel='rbf', cache_size=1000)
    #clf.probability = True
    clf.fit(X, y)
    #probabilities = clf.predict_proba(X)  # values between [0,1]
    medical_test_scores.append([clf.score(X, y),
                                roc_auc_score(y, clf.decision_function(X))])

for i, test in enumerate(medical_tests):
    print("\n" + test + "\n average accuracy: " + str(medical_test_scores[i][0]) +
          "\n roc_auc score: " + str(medical_test_scores[i][1]))


# Train classifier to predict sepsis event
X = train_features_flat
y = train_labels['LABEL_Sepsis']
clf = svm.SVC(kernel='rbf', cache_size=1000)
clf.fit(X, y)
print("\nSepsis prediction\n average accuracy: " + str(clf.score(X, y)) +
      "\n roc_auc score: " + str(roc_auc_score(y, clf.decision_function(X))))

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




