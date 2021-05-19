import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import repeat
import multiprocessing

# keep this stuff global
extraction_net = models.resnet101(pretrained=True)


def process_image(img_tuple, features):
    i, data = img_tuple
    print(i)
    out = extraction_net(data[0])
    features[i] = out.data.numpy()[0]


def extract_features():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)

    dataset_size = len(image_dataset)
    class_name = image_dataset.classes

    features_dict = multiprocessing.Manager().dict()
    n_workers = multiprocessing.cpu_count() - 1
    chunksize, extra = divmod(dataset_size, n_workers * 4)  # good chunksize heuristic
    if extra:
        chunksize += 1
    print("dataset_size: ", dataset_size)
    print("n_workers: ", n_workers)
    print("chunksize: ", chunksize)

    with multiprocessing.Pool(n_workers) as P:
        P.starmap(process_image, zip(enumerate(dataloader), repeat(features_dict)), chunksize=chunksize)
        P.close()
        P.join()

    final_features = []
    for i in range(len(features_dict)):
        final_features.append(features_dict[i])

    return pd.DataFrame(final_features)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def test_reproducibility():
    print("Testing reproducibility of feature extraction")
    np.random.seed(42)
    torch.manual_seed(42)
    test_features_df = extract_features()
    np.random.seed(42)
    torch.manual_seed(42)
    test_features_df_2 = extract_features()
    print(test_features_df)
    print(test_features_df_2)
    print("Reproducible?", test_features_df.equals(test_features_df_2))


if __name__ == "__main__":
    TEST_REPROD = False

    if TEST_REPROD:
        test_reproducibility()

    else:
        np.random.seed(42)
        torch.manual_seed(42)
        final_features_df = extract_features()
        final_features_df.to_csv('data/train_image_features.csv', index=True, header=False)


#
# # Get a batch of training data
# image, classes = next(iter(dataloader))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(image)
#
# imshow(out)
