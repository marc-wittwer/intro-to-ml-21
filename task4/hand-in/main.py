import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time


num_epochs = 10  # each epoch is one pass over the whole dataset
batch_size = 128  # how many samples per training step (32,64,128)
num_workers = 7  # how many CPU cores to use

n_features = 1000  # How many features per image

# Specify some file names
img_features_file = "data/image_features.csv"
train_triplets_file = "data/train_triplets.txt"
test_triplets_file = "data/test_triplets.txt"


class FoodTriplets(Dataset):
    """
    Class to load food triplets. Individual items consist of:
     - x: concatenated image features of a triplet (ABC)
     - y: corresponding label (1: B is more similar, 0: C is more similar, None: no label)
    """
    def __init__(self, features_file, triplets_file, is_labelled_data=False):
        print("initializing " + str(triplets_file) + " dataset...")
        self.img_features = pd.read_csv(features_file, header=None, index_col=0)
        self.triplets = pd.read_csv(triplets_file, sep=" ", header=None).to_numpy()
        self.is_labelled_data = is_labelled_data
        self.labels = None

        if self.is_labelled_data:
            # For each triplet ABC add a triplet ACB
            # Also add corresponding labels
            dim = self.triplets.shape
            extended_triplets = np.zeros((2*dim[0], dim[1]), dtype=np.int32)
            self.labels = np.zeros(2*dim[0], dtype=bool)
            idx = 0
            for triplet in self.triplets:
                extended_triplets[idx, :] = triplet
                extended_triplets[idx + 1, :] = [triplet[0], triplet[2], triplet[1]]
                self.labels[idx] = 1  # label at idx+1 is already initialized to 0
                idx += 2

            self.triplets = extended_triplets

        print("done")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        a, b, c = triplet[0], triplet[1], triplet[2]
        a_features = self.img_features.loc[[a]].to_numpy(dtype=np.float32)
        b_features = self.img_features.loc[[b]].to_numpy(dtype=np.float32)
        c_features = self.img_features.loc[[c]].to_numpy(dtype=np.float32)
        features = np.squeeze(np.concatenate((a_features, b_features, c_features), axis=1))
        label = 0  # dummy label
        if self.is_labelled_data:
            label = np.array([self.labels[idx]])
        return {"x": features, "y": label}


def get_model(device):
    return SimpleNetwork().to(device)


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        nin = 3*n_features
        n1 = n_features
        n2 = int(n_features/2)
        n3 = int(n_features/8)
        nout = 1

        p_dropout1 = 0.5
        p_dropout2 = 0.5
        p_dropout3 = 0.4

        self.classifier = nn.Sequential(
            nn.Linear(in_features=nin, out_features=n1),
            nn.BatchNorm1d(n1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout1, inplace=False),
            nn.Linear(in_features=n1, out_features=n2),
            nn.BatchNorm1d(n2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout2, inplace=False),
            nn.Linear(in_features=n2, out_features=n3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n3),
            nn.Dropout(p=p_dropout3, inplace=False),
            nn.Linear(in_features=n3, out_features=nout),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        logits = self.classifier(x_in)
        return logits


def experiment(num_epochs, trainloader, device, model, optimizer, scheduler):
    train_results = {}
    test_results = {}
    # Initial test error
    loss, acc, time = test(device, model)
    print(f'Upon initialization. [Test] \t Time {time.avg:.2f} \
            Loss {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
    test_results[0] = (loss, acc, time)

    for epoch in range(1, num_epochs+1):
        loss, acc, time = train(trainloader, device, model, optimizer, scheduler)
        print(f'Epoch {epoch}. [Train] \t Time {time.sum:.2f} Loss \
                {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
        train_results[epoch] = (loss.avg, acc.avg, time.avg)

        if not (epoch % 2):
          loss, acc, time = test(device, model)
          print(f'Epoch {epoch}. [Test] \t Time {time.sum:.2f} Loss \
                {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
          test_results[epoch] = (loss.avg, acc.avg, time.avg)


def train(trainloader, device, model, optimizer, scheduler):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.train()

    for i, data, in enumerate(trainloader, 1):
        # Accounting
        end = time.time()

        # get the inputs; data is a list of [inputs, labels]
        inputs = data["x"].float()
        labels = data["y"].float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        bs = inputs.size(0)
        # zero the parameter gradients
        optimizer.zero_grad()  # all the tensors have .grad attribute
        # forward propagation
        logits = model(inputs) # forward propagation
        loss = criterion(logits, labels) # computing the loss for predictions
        # Backward propagation
        loss.backward()  # backpropgation
        # Optimization step.
        optimizer.step()  # applying an optimization step

        # Accounting
        acc = ((torch.round(logits) == labels).sum().float() / bs).float()
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(time.time() - end)

    scheduler.step()
    return loss_, acc_, time_


def test(device, model):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.eval()

    for i, data, in enumerate(train_testloader, 1):
        # Accounting
        end = time.time()

        inputs = data["x"].float()
        labels = data["y"].float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        bs = inputs.size(0)

        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits, labels)
            acc = ((torch.round(logits) == labels).sum().float() / bs).float()

            # Accounting
            loss_.update(loss.mean().item(), bs)
            acc_.update(acc.mean().item(), bs)
            time_.update(time.time() - end)

    return loss_, acc_, time_


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    # initialize datasets
    train_data = FoodTriplets(img_features_file, train_triplets_file, is_labelled_data=True)
    test_data = FoodTriplets(img_features_file, test_triplets_file, is_labelled_data=False)

    # Split train set into train and test set to assess accuracy on unused set
    l_train = len(train_data)
    val_size = int(0 * l_train)+1
    indices = list(range(l_train))
    np.random.shuffle(indices)
    val_indices, t_indices = indices[:val_size], indices[val_size:]

    train_trainloader = DataLoader(torch.utils.data.Subset(train_data, t_indices),
                                   batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_testloader = DataLoader(torch.utils.data.Subset(train_data, val_indices),
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    experiment(num_epochs=num_epochs,
               trainloader=train_trainloader,
               device=device,
               model=model,
               optimizer=optimizer,
               scheduler=lr_scheduler)

    # predict on test_data
    predictions = np.zeros(len(test_data))
    print("predicting on test_data...")
    l_test = len(test_data)
    start_t = time.time()

    # Set model to evaluating mode
    model.eval()

    for i, data in enumerate(test_dataloader, 0):
        x = data["x"].float()

        x = x.to(device)
        predictions[i] = model(x)
        if i % 1000 == 0:
            print(f"Predicted: {i}/{l_test}, Time elapsed: {timer(start_t, time.time())}")

    predictions_DF = pd.DataFrame(predictions)
    predictions_rounded = predictions_DF.round(0)
    print("done")

    res = pd.DataFrame(predictions_rounded).astype(int)
    print("saving predictions...")
    res.to_csv('data/predictions.csv', index=False, header=False)
    print("done")
