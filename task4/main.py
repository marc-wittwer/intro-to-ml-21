import os
import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
import math
import collections


def fifty_fifty():
    return np.random.random() < 0.5


class FoodTripletsData:
    def __init__(self, img_features, trp_path, batch_size, shuffle=False, train=False, extend=True, n_rbf=0):
        print(f"Initializing dataset: {trp_path}")
        self.train = train  # if True: balance dataset + create labels
        self.extend = extend  # if True: extend triplets with inverted labels
        self.n_rbf = n_rbf  # number of components to approximate rbf kernel (0 = don't use rbf kernel)

        # load features
        self.img_features = img_features
        self.triplets = pd.read_csv(trp_path, sep=" ", header=None).to_numpy()
        self.labels = None

        # balance (and possibly extend) training set
        if train:
            self.triplets, self.labels = self._balance_triplets(self.triplets)

        # shuffle triplets & labels accordingly
        if shuffle:
            self.shuffle()

        if batch_size < 1:
            self.batch_size = self.triplets.shape[0]
            self.n_batches = 1
        else:
            self.batch_size = batch_size
            self.n_batches = int(math.ceil(self.triplets.shape[0] / self.batch_size))

        self.n_features = self.img_features.shape[1]

        # create sampler to approximate rbf kernel feature transform (for nonlinear classification)
        if self.n_rbf > 0:
            self.rbf_sampler = RBFSampler(gamma=1, n_components=self.n_rbf)

        print(f"Done. Dataset contains {self.triplets.shape[0]} triplets.")

    def _balance_triplets(self, triplets):
        """
        Make sure labels for train set are balanced. If extend=False, randomly swap half of the triplets' BC and label,
        if extend=True, 'duplicate' each triplet with switched BC and label.
        """
        dim = triplets.shape
        balanced_triplets = np.zeros(((1 + self.extend) * dim[0], dim[1]), dtype=np.int32)
        labels = np.zeros((1 + self.extend) * dim[0], dtype=np.bool)
        idx = 0

        if self.extend:
            for triplet in triplets:
                # add a second triplet with swapped BC and inverted label
                balanced_triplets[idx, :] = triplet
                labels[idx] = 1
                balanced_triplets[idx + 1, :] = [triplet[0], triplet[2], triplet[1]]
                # label is already initialized to 0
                idx += 2

            pass
        else:
            for triplet in triplets:
                # randomly (50% chance) decide to swap BC and invert label
                if fifty_fifty():
                    balanced_triplets[idx, :] = triplet
                    labels[idx] = 1
                else:
                    balanced_triplets[idx, :] = [triplet[0], triplet[2], triplet[1]]
                    # label is already initialized to 0
                idx += 1
        return balanced_triplets, labels

    def __len__(self):
        return self.triplets.shape[0]

    def shuffle(self):
        """
        Randomly permute triplets and labels accordingly
        """
        assert len(self.triplets) == len(self.labels)
        p_indices = np.random.permutation(len(self.triplets))
        self.triplets = self.triplets[p_indices]
        self.labels = self.labels[p_indices]

    def __getitem__(self, idx):
        x, y = self._get_item(idx)
        if self.n_rbf > 0:
            x = self.rbf_sampler.fit_transform(x)
        return x, y

    def _get_item(self, idx):
        triplet = self.triplets[idx]
        a, b, c = triplet[0], triplet[1], triplet[2]
        a_features = self.img_features.loc[[a]].to_numpy(dtype=np.float32)
        b_features = self.img_features.loc[[b]].to_numpy(dtype=np.float32)
        c_features = self.img_features.loc[[c]].to_numpy(dtype=np.float32)
        features = np.squeeze(np.concatenate((a_features, b_features, c_features), axis=1))
        label = 0
        if self.train:
            label = np.array([self.labels[idx]])
        return features, label

    def get_batch(self, i):
        # get indices of items to collect in i-th batch
        indices = range(i * self.batch_size, (i + 1) * self.batch_size)
        if i+1 == self.n_batches:
            # final batch probably contains fewer items
            indices = range(i * self.batch_size, self.triplets.shape[0])

        x_batch = np.zeros((len(indices), 3 * self.n_features), dtype=np.float32)
        y_batch = np.zeros(len(indices), dtype=np.bool)

        k = 0
        for idx in indices:
            x, y = self._get_item(idx)
            x_batch[k] = x
            y_batch[k] = y
            k += 1

        # fit rbf kernel if desired
        if self.n_rbf > 0:
            x_batch = self.rbf_sampler.fit_transform(x_batch)

        return x_batch, y_batch

    def get_split_batch(self, i, split=0.8):
        """
        Get mini-batch split in train and test part. (split=0.8 means 80% train, 20% test).
        """
        x_batch, y_batch = self.get_batch(i)
        cut = int(len(x_batch) * split)
        x_train, x_test = x_batch[:cut], x_batch[cut:]
        y_train, y_test = y_batch[:cut], y_batch[cut:]
        return x_train, x_test, y_train, y_test


def compute_accuracy(predictions, ground_truth):
    compared = np.equal(predictions, ground_truth)
    n_correct = np.sum(compared, axis=0)
    accuracy = n_correct / predictions.shape[0]
    return accuracy


def train_sgd_incremental(dataset, n_epochs=1, split=0.8, tol=0):
    sgd = SGDClassifier()

    prev_avg_acc = 0
    for epoch in range(n_epochs):
        avg_acc = 0
        te = time.time()
        for i in range(dataset.n_batches):
            ts = time.time()
            x_train, x_test, y_train, y_test = dataset.get_split_batch(i, split=split)
            sgd.partial_fit(x_train, y_train, classes=np.array([True, False]))
            y_pred = sgd.predict(x_test)
            accuracy = compute_accuracy(y_pred, y_test)
            avg_acc += accuracy * (len(y_train) + len(y_pred))  #  track accuracy over epoch, account for varying batch sizes
            print(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{dataset.n_batches}, "
                  f"Test Accuracy: {accuracy:.3f}, Time: {timer(ts, time.time())}")

        # compute average accuracy of this epoch, terminate if no improvement happens
        avg_acc = avg_acc / len(dataset)
        acc_diff = avg_acc - prev_avg_acc
        print(f"   Epoch {epoch+1} complete. Average accuracy: {avg_acc} (+{acc_diff}), Time: {timer(te, time.time())}")
        if (acc_diff < tol) and (avg_acc > 0.8):
            print(f"Terminating training early. Avg Acc: {avg_acc}, Epoch: {epoch+1}")
            break
        prev_avg_acc = avg_acc
        dataset.shuffle()
    return sgd


def predict(classifier, dataset):
    bs = dataset.batch_size
    predictions = np.zeros(len(dataset))
    for i in range(dataset.n_batches):
        x, _ = dataset.get_batch(i)
        pred = classifier.predict(x)
        predictions[(i * bs):(i * bs + len(pred))] = pred
    return predictions


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


# stack all feature triplets
def build_features_and_labels(img_features, train_trp, test_trp):
    n_train = len(train_trp)  # number of train triplets
    n_test = len(test_trp)  # number of test triplets
    n_features = img_features.shape[1]  # number of features per image
    train_x = np.zeros((2*n_train, 3*n_features), dtype=float)
    train_y = np.zeros(2*n_train, dtype=int)
    test_x = np.zeros((n_test, 3*n_features), dtype=float)

    # compose train features and labels
    print("composing train set...")
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
    print("composing test set...")
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


# train clf using grid search on complete train set
def train_classifier(train_features, train_labels, n_jobs=1, cv=5):

    cat_features = np.ones(train_features.shape[1], dtype=bool)
    if len(cat_features) == 223*3:
        # padding is no categorical feature (if even included)
        for i in range(3):
            print("features include padding count.")
            cat_features[223*(i+1)-1] = False
            print(223*(i+1)-1)

    grid_search_params = [
        {'clf': HistGradientBoostingClassifier(random_state=42),
         'params': {'clf__l2_regularization': [3],
                    'clf__max_leaf_nodes': [31],
                    'clf__min_samples_leaf': [30],
                    'clf__learning_rate': [0.05],
                    'clf__categorical_features': [cat_features],
                    'clf__max_iter': [500, 700, 1000]}
         }
    ]

    results = []
    best_estimators = []
    #
    for params in grid_search_params:
        pipeline = Pipeline(steps=[('clf', params['clf'])])

        grid_search = GridSearchCV(pipeline, param_grid=params['params'], scoring='accuracy', n_jobs=n_jobs, verbose=3,
                                   cv=cv)

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


def main_sgd():
    train_test_split = 0.8  # fraction of train set
    batch_size = 5000  # samples per partial fit
    n_epochs = 10  # passes over the complete dataset
    n_components_rbf = 0  # number of samples to approximate rbf kernel.
    # higher means more accurate but more computation.
    # n_components >> n_features for a useful approx.

    np.random.seed(42)
    print("start: " + str(time.ctime()))
    t = time.time()

    # load features
    image_features_path = "data/train_image_features_vgg16bn.csv"
    train_triplets_path = "data/train_triplets.txt"
    test_triplets_path = "data/test_triplets.txt"

    # load image features (avoid loading twice, it's a big file, datasets can share it)
    print("Loading image features...")
    image_features = pd.read_csv(image_features_path, header=None, index_col=0)
    image_features.drop("<pad>", axis=1)  # don't count padding as feature

    # train classifier
    train_dataset = FoodTripletsData(image_features, train_triplets_path,
                                     batch_size=batch_size, shuffle=True,
                                     train=True, extend=False,
                                     n_rbf=n_components_rbf)
    print("training classifier...")
    clf = train_sgd_incremental(train_dataset, n_epochs=n_epochs, split=train_test_split,
                                tol=0)

    # predict on test set
    test_dataset = FoodTripletsData(image_features, test_triplets_path,
                                    batch_size=batch_size, shuffle=False,
                                    train=False, n_rbf=n_components_rbf)
    print("predicting on test set...")
    y_test_predictions = predict(clf, test_dataset)

    # #  Save predictions to file
    predictions_df = pd.DataFrame(y_test_predictions).astype(int)
    predictions_df.to_csv('data/predictions.csv', index=False, header=False)
    print(f"Fitting and prediction took {timer(t, time.time())}")


def get_feature_counts():
    ing_feature_counts_path = "data/ing_feature_counts.csv"
    ing_features_path = "data/ing_features.csv"
    ing_vocab_path = "data/ing_vocab.csv"
    if os.path.isfile(ing_feature_counts_path):
        print("Loading existing feature counts...")
        ing_feature_counts = pd.read_csv(ing_feature_counts_path, index_col=0, header=0)

    else:
        print("No feature counts found. Computing them now...")
        ing_features = pd.read_csv(ing_features_path, index_col=0, header=None)
        ing_vocabulary = pd.read_csv(ing_vocab_path, header=None)

        ing_feature_counts = ing_features.apply(collections.Counter, axis=1)
        ing_feature_counts = pd.DataFrame.from_records(ing_feature_counts).fillna(value=0)
        ing_feature_counts = ing_feature_counts.reindex(sorted(ing_feature_counts.columns), axis=1)  # sort columns
        for i in range(len(ing_feature_counts.columns)):
            col = ing_feature_counts.columns[i]
            new_name = ing_vocabulary.loc[col, 0]
            ing_feature_counts.rename(columns={col: new_name}, inplace=True)
        ing_feature_counts.to_csv(ing_feature_counts_path, index=True, header=True)

    print(ing_feature_counts)
    return ing_feature_counts


def main_ing():
    n_jobs = 3
    cv = 2
    np.random.seed(42)

    # load triplets
    train_triplets_path = "data/train_triplets.txt"
    test_triplets_path = "data/test_triplets.txt"

    # load image features (ingredient counts)
    features = get_feature_counts()

    # build train set
    train_set = FoodTripletsData(features, train_triplets_path, -1, shuffle=True,
                                 train=True, extend=False, n_rbf=0)
    train_x, train_y = train_set.get_batch(0)

    # train classifier (gradient boosted trees)
    clf = train_classifier(train_x, train_y, n_jobs=n_jobs, cv=cv)

    # build test set
    test_set = FoodTripletsData(features, test_triplets_path, -1, shuffle=False,
                                train=False, extend=False, n_rbf=0)
    test_x, _ = test_set.get_batch(0)

    # predict on test set
    predictions = clf.predict(test_x)
    predictions_df = pd.DataFrame(predictions).astype(int)
    predictions_df.to_csv('data/predictions.csv', index=False, header=False)


if __name__ == "__main__":
    # main_sgd()
    main_ing()
