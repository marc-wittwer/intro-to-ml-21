import os
import pandas as pd
from torchvision import datasets, models, transforms
import torch
import numpy as np
import time


def extract_features():
    data_transform = transforms.Compose([
        transforms.Resize((350)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'data'
    image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)

    features_dict = {}
    start_time = time.time()

    for img_tuple in enumerate(dataloader):
        i, data = img_tuple

        image = data[0]
        image = image.to(device)

        out = extraction_net(image)
        out = out.to('cpu')

        features_dict[i] = out.data.numpy()[0]

        if i % 100 == 0:
            print(f"Img: {i} \t Time elapsed: {timer(start_time, time.time())}")

    final_features = []
    for i in range(len(features_dict)):
        final_features.append(features_dict[i])

    return pd.DataFrame(final_features)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extraction_net = models.mobilenet_v3_small(pretrained=True, progress=True)
    extraction_net.to(device)
    extraction_net.eval()

    final_features_df = extract_features()
    final_features_df.to_csv('data/image_features_mobileNetv3small350x350.csv', index=True, header=False)
