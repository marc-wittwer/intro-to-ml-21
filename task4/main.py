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


def fifty_fifty():
    if np.random.random() < 0.5:
        return True
    else:
        return False


class FoodTripletsData:
    def __init__(self, img_features, trp_path, batch_size, shuffle=False, train=False, extend=True):
        print(f"Initializing dataset: {trp_path}")
        self.train = train  # if True: balance dataset + create labels
        self.extend = extend  # if True: extend triplets with inverted labels

        # load features
        self.img_features = img_features
        self.triplets = pd.read_csv(trp_path, sep=" ", header=None).to_numpy()
        self.labels = None

        if train:
            self.triplets, self.labels = self._balance_triplets(self.triplets)

        if shuffle:
            self._shuffle_dataset()

        self.batch_size = batch_size
        self.n_batches = int(math.ceil(self.triplets.shape[0] / self.batch_size))
        self.n_features = self.img_features.shape[1]

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

    def _shuffle_dataset(self):
        """
        Randomly permute triplets and labels accordingly
        """
        assert len(self.triplets) == len(self.labels)
        p_indices = np.random.permutation(len(self.triplets))
        self.triplets = self.triplets[p_indices]
        self.labels = self.labels[p_indices]

    def __getitem__(self, idx):
        return self._get_item(idx)

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
        return {"x": features, "y": label}

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
            item = self._get_item(idx)
            x_batch[k] = item["x"]
            y_batch[k] = item["y"]
            k += 1

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
                  f"Test Accuracy: {accuracy}, Time: {timer(ts, time.time())}")

        # compute average accuracy of this epoch, terminate if no improvement happens
        avg_acc = avg_acc / len(dataset)
        print(f"    Epoch {epoch+1} complete. Average accuracy: {avg_acc}, Time: {timer(te, time.time())}")
        if ((avg_acc - prev_avg_acc) < tol) and (avg_acc > 0.8):
            print(f"Terminating training early. Avg Acc: {avg_acc}, Epoch: {epoch+1}")
            break
    return sgd


def predict(classifier, dataset):
    bs = dataset.batch_size
    predictions = np.zeros(len(test_dataset))
    for i in range(dataset.n_batches):
        x, _ = dataset.get_batch(i)
        pred = classifier.predict(x)
        predictions[(i * batch_size):(i * batch_size + len(pred))] = pred
    return predictions


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


# old function, currently unused (stack all feature triplets)
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


# old function, currently unused (train clf using grid search on complete train set)
def train_classifier(train_features, train_labels):
    grid_search_params = [
        {'clf': HistGradientBoostingClassifier(max_iter=100, random_state=42),
         'params': {'clf__l2_regularization': [0],
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
                                   cv=2)

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
    train_test_split = 0.8  # fraction of train set
    batch_size = 5000
    n_epochs = 10

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

    # train classifier
    train_dataset = FoodTripletsData(image_features, train_triplets_path,
                                     batch_size=batch_size,
                                     shuffle=True, train=True, extend=True)
    print("training classifier...")
    clf = train_sgd_incremental(train_dataset, n_epochs=n_epochs, split=train_test_split)

    # predict on test set
    test_dataset = FoodTripletsData(image_features, test_triplets_path,
                                    batch_size=batch_size,
                                    shuffle=False, train=False)
    print("predicting on test set...")
    y_test_predictions = predict(clf, test_dataset)

    # #  Save predictions to file
    predictions_df = pd.DataFrame(y_test_predictions).astype(int)
    predictions_df.to_csv('data/predictions.csv', index=False, header=False)
    print(f"Fitting and prediction took {timer(t, time.time())}")
