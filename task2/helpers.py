import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, r2_score
import matplotlib.pyplot as plt


def generate_model_report(y_actual, y_predicted):
    print("Accuracy = ", accuracy_score(y_actual, y_predicted))
    print("Precision = ", precision_score(y_actual, y_predicted))
    print("Recall = ", recall_score(y_actual, y_predicted))
    print("F1 Score = ", f1_score(y_actual, y_predicted))
    pass


def generate_auc_roc_curve(clsf, x_test, y_test):
    y_pred_proba = clsf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC ROC Curve with Area Under the curve =" + str(auc))
    plt.legend(loc=4)
    plt.show()
    pass


def inspect_nan_statistics_per_feature(data):
    """
    count number of NaNs in each column for each patient. Summarize and print stats of those counts.
    """
    n_timesteps = 12
    n_patients = int(data.shape[0] / n_timesteps)
    print("There are " + str(n_patients) + " patients in this dataframe")
    data = data.notna()  # boolean array (1 if value is not NaN, 0 if Nan)
    nan_count_per_patient = np.zeros((n_patients, data.shape[1]))

    for i in range(n_patients):
        patient_data = data.iloc[n_timesteps*i:(n_timesteps*(i+1)), :]  # select rows corresponding to patient
        nan_count_per_patient[i] = np.sum(patient_data, axis=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        nan_count_per_patient = pd.DataFrame(nan_count_per_patient)
        nan_count_per_patient.columns = train_features.columns
        stats = nan_count_per_patient.describe()
        print(stats)


if __name__ == "__main__":
    train_features = pd.read_csv('data/train_features.csv')
    inspect_nan_statistics_per_feature(train_features)
