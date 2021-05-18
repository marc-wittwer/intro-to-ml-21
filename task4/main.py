import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier


def build_features_and_labels(img_features, train_trp, test_trp):
    n_train = len(train_trp)  # number of train triplets
    n_test = len(test_trp)  # number of test triplets
    n_features = img_features.shape[1]  # number of features per image
    train_x = np.zeros((2*n_train, 3*n_features))
    train_y = np.zeros(2*n_train)
    test_x = np.zeros((2*n_test, 3*n_features))

    # compose train features and labels
    idx = 0
    for triplet in train_trp.to_numpy():
        # a: anchor, b: positive, c: negative
        a, b, c = triplet[0], triplet[1], triplet[2]  # image names = feature indices

        # extract corresponding features from img_features
        a_features = img_features.loc[[a]].to_numpy()
        b_features = img_features.loc[[b]].to_numpy()
        c_features = img_features.loc[[c]].to_numpy()

        # build train_x array (train features) and corresponding train_y (train labels)
        train_x[idx, 0:n_features] = a_features
        train_x[idx, n_features:2*n_features] = b_features
        train_x[idx, 2*n_features:3*n_features] = c_features
        train_y[idx] = 1
        # add an extra row with swapped a and b for more train data and class balance
        idx += 1
        train_x[idx, 0:n_features] = a_features
        train_x[idx, n_features:2 * n_features] = c_features
        train_x[idx, 2 * n_features:3 * n_features] = b_features
        # train_y[idx] = 0  # no need to set to 0, it is already initialized this way
        idx += 1

    # compose test features
    idx = 0
    for triplet in test_trp.to_numpy():
        # a: anchor, b: positive, c: negative
        a, b, c = triplet[0], triplet[1], triplet[2]  # image names = feature indices

        # extract corresponding features from img_features
        a_features = img_features.loc[[a]].to_numpy()
        b_features = img_features.loc[[b]].to_numpy()
        c_features = img_features.loc[[c]].to_numpy()

        # build test_x array (test features)
        test_x[idx, 0:n_features] = a_features
        test_x[idx, n_features:2 * n_features] = b_features
        test_x[idx, 2 * n_features:3 * n_features] = c_features
        idx += 1

    return train_x, train_y, test_x


def train_classifier(train_features, train_labels):
    grid_search_params = [
        {'clf': HistGradientBoostingClassifier(max_iter=100, random_state=42),
         'params': {'clf__l2_regularization': [1],
                    'clf__max_leaf_nodes': [31],
                    'clf__min_samples_leaf': [20],
                    'clf__learning_rate': [0.1]
                    }
         }
    ]

    results = []
    best_estimators = []
    #
    for params in grid_search_params:
        pipeline = Pipeline(steps=[('clf', params['clf'])])

        grid_search = GridSearchCV(pipeline, param_grid=params['params'], scoring='accuracy', n_jobs=n_jobs, verbose=3,
                                   cv=None)

        grid_search.fit(train_features, train_labels)

        # keep track of results
        df = pd.DataFrame({'classifier': str(params['clf']),
                           'mean_score': grid_search.cv_results_["mean_test_score"],
                           'fit_time': grid_search.cv_results_["mean_fit_time"]})
        for key in params["params"]:
            if len(params["params"][key]) > 1:
                df[key] = grid_search.cv_results_["param_" + key]
        results.append(df)

        # keep track of best estimator
        best_estimators.append((grid_search.best_estimator_, grid_search.best_score_))

        print('Best parameters:', grid_search.best_params_, 'Accuracy:', grid_search.best_score_)

    # Display results for inspection
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        for i, res in enumerate(results):
            print("\n")
            print(res)
            print("Best estimator (with a score of " + str(best_estimators[i][1]) + "): ")
            print(best_estimators[i][0])

    # Predict labels for test data with best classifier
    best_estimators.sort(key=lambda x: x[1], reverse=True)  # sort by score (descending)
    best_clf = best_estimators[0][0]
    return best_clf


if __name__ == "__main__":
    n_jobs = 3
    np.random.seed(42)

    # load features
    image_features = pd.read_csv("data/image_features.csv", header=None, index_col=0)
    train_triplets = pd.read_csv("data/train_triplets.txt", sep=" ", header=None)
    test_triplets = pd.read_csv("data/test_triplets.txt", sep=" ", header=None)

    image_features.info()
    train_triplets.info()

    # remove triplets containing features that have not been extracted yet
    train_triplets = train_triplets[train_triplets.max(axis=1) < len(image_features)]
    test_triplets = test_triplets[test_triplets.max(axis=1) < len(image_features)]
    print("remaining number of triplets: ")
    print("train:", len(train_triplets))
    print("test:", len(test_triplets))

    # create train + test triplets (concatenate features) + labels
    train_features_np, train_labels_np, test_features_np = build_features_and_labels(image_features, train_triplets, test_triplets)

    # train classifier
    clf = train_classifier(train_features_np, train_labels_np)

    # # predict on test set
    # y_test_predictions = pd.DataFrame(clf.predict(test_features_np))
    #
    # # #  Save predictions to file
    # y_test_predictions.to_csv('data/predictions.csv', index=False, header=False)
