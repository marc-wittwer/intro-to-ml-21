import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

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
image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=2)

dataset_size = len(image_dataset)
class_name = image_dataset.classes

extraction_net = models.resnet101(pretrained=True)

feature_df = pd.DataFrame()

for i in range(dataset_size*0+1000):
    print(i)
    image, classes = next(iter(dataloader))
    out = extraction_net(image)
    feature_df = feature_df.append(pd.DataFrame(out.data.numpy(), index=[f'{i:05}']))


feature_df.to_csv('data/image_features.csv', index=True, header=False)



# To display the image
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
#
# # Get a batch of training data
# image, classes = next(iter(dataloader))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(image)
#
# imshow(out)
