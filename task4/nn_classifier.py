import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import numpy as np
import time


class FoodTriplets(Dataset):
    """
    Class to load food triplets. Individual items consist of:
     - x: concatenated image features of a triplet (ABC)
     - y: corresponding label (1: B is more similar, 0: C is more similar, None: no label)
    """
    def __init__(self, features_file, triplets_file, is_labelled_data=False):
        print("initializing " + str(triplets_file) + " dataset...")
        self.img_features = pd.read_csv("data/" + features_file, header=None, index_col=0)
        self.triplets = pd.read_csv("data/" + triplets_file, sep=" ", header=None).to_numpy()
        self.is_labelled_data = is_labelled_data
        self.labels = None

        if self.is_labelled_data:
            # For each triplet ABC add a triplet ACB
            # Also add corresponding labels
            dim = self.triplets.shape
            extended_triplets = np.zeros((2*dim[0], dim[1]), dtype=float)
            self.labels = np.zeros(2*dim[0], dtype=int)
            idx = 0
            for triplet in self.triplets:
                extended_triplets[idx, :] = triplet
                extended_triplets[idx + 1, :] = [triplet[0], triplet[2], triplet[1]]
                self.labels[idx] = 1  # label at idx+1 is already initialized to 0
                idx += 2
        print("done")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        a, b, c = triplet[0], triplet[1], triplet[2]
        a_features = self.img_features.loc[[a]].to_numpy()
        b_features = self.img_features.loc[[b]].to_numpy()
        c_features = self.img_features.loc[[c]].to_numpy()
        features = np.concatenate((a_features, b_features, c_features))
        label = None
        if self.is_labelled_data:
            label = self.labels[idx]
        return {"x": features, "y": label}


def get_simple_nn():

    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.8, inplace=False),
        nn.Linear(in_features=12288, out_features=6144),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.8, inplace=False),
        nn.Linear(in_features=6144, out_features=1)
    )


if __name__ == "__main__":

    # Specify some file names
    img_features_file = "train_image_features_vgg16bn.csv"
    train_triplets_file = "train_triplets.txt"
    test_triplets_file = "test_triplets.txt"

    train_data = FoodTriplets(img_features_file, train_triplets_file, is_labelled_data=True)
    test_data = FoodTriplets(img_features_file, test_triplets_file, is_labelled_data=False)

    for i in range(10):
        train_sample = train_data[i]
        test_sample = test_data[i]
        print(i)
        print("train", train_sample["x"].shape, train_sample["y"])
        print("test", test_sample["x"].shape, test_sample["y"])

    model = get_simple_nn()

