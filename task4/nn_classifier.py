import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import time
from pathlib import Path


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
            extended_triplets = np.zeros((2*dim[0], dim[1]), dtype=np.int32)
            self.labels = np.zeros(2*dim[0], dtype=np.bool)
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
        a_features = self.img_features.loc[[a]].to_numpy(dtype=np.float32)
        b_features = self.img_features.loc[[b]].to_numpy(dtype=np.float32)
        c_features = self.img_features.loc[[c]].to_numpy(dtype=np.float32)
        features = np.concatenate((a_features, b_features, c_features), axis=1)
        label = 0  # dummy label
        if self.is_labelled_data:
            label = np.array([[self.labels[idx]]])
        return {"x": features, "y": label}


def get_simple_nn():
    net = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.8, inplace=False),
        nn.Linear(in_features=12288, out_features=6144),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.8, inplace=False),
        nn.Linear(in_features=6144, out_features=1)
    )
    return net


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


if __name__ == "__main__":
    n_epochs = 2  # each epoch is one pass over the whole dataset
    batch_size = 4  # how many samples per training step
    num_workers = 3  # how many CPU cores to use

    # Specify some file names
    img_features_file = "train_image_features_vgg16bn.csv"
    train_triplets_file = "train_triplets.txt"
    test_triplets_file = "test_triplets.txt"

    # initialize datasets and dataloaders
    train_data = FoodTriplets(img_features_file, train_triplets_file, is_labelled_data=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data = FoodTriplets(img_features_file, test_triplets_file, is_labelled_data=False)
    test_dataloader = DataLoader(test_data, batch_size=None, shuffle=False, num_workers=num_workers)

    # define model, loss function and optimizer
    model = get_simple_nn().float()
    loss_function = nn.BCEWithLogitsLoss()  # combines sigmoid with binary cross entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_batches = math.ceil(len(train_data) / batch_size)

    # train model
    print(f"training started at {time.ctime()}")
    start_t = time.time()
    print_frequency = 1
    for epoch in range(n_epochs):
        running_loss = 0
        for i, data in enumerate(train_dataloader):
            inputs = data["x"].float()
            labels = data["y"].float()
            optimizer.zero_grad()  # set gradients to zero

            # forward pass, backprop + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_frequency == 0:
                print(f"Epoch: {epoch+1}/{n_epochs}, Batch: {i}/{n_batches}, loss: {running_loss/print_frequency}, "
                      f"Time elapsed: {timer(start_t, time.time())}")
                running_loss = 0

    print("finished training")

    # save model parameters
    Path("model").mkdir(parents=True, exist_ok=True)  # create folder if necessary
    torch.save(model.state_dict(), "model/params")
    # load: model.load_state_dict(torch.load("model/params"))

    # predict on test_data
    model_pred = nn.Sequential(model, nn.Softmax(dim=0))  # add softmax to get 1/0 output
    predictions = np.zeros(len(test_data))
    print("predicting on test_data...")
    l_test = len(test_data)
    for i, data in enumerate(test_dataloader, 0):
        x = data["x"].float()
        predictions[i] = model_pred(x)
        if i % 1000 == 0:
            print(f"Predicted: {i}/{l_test}, Time elapsed: {timer(start_t, time.time())}")
    predictions = pd.DataFrame(predictions).astype(int)
    print("saving predictions...")
    predictions.to_csv('data/predictions.csv', index=False, header=False)
    print("done")
