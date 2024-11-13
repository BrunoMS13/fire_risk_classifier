import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.io import read_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        target_transform=None,
        task="classification",
    ):

        self.img_labels = pd.read_csv(annotations_file)
        classes = sorted(self.img_labels.iloc[:, 1].unique())

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.task = task.lower()

        self.classes = [each_class for each_class in classes]
        self.class2idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx2class = {i: self.classes[i] for i in range(len(self.classes))}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = read_image(img_path).float()
        if self.task == "classification":
            label = self.class2idx[self.img_labels.iloc[idx, 1]]
        else:
            raise ValueError('Task can only be "classification"')

        if self.transform:
            image = self.transform(image)

        if label >= 0 and self.target_transform:
            label = self.target_transform(label)
        return image, label

    def show(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = read_image(img_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")
        plt.title(self.img_labels.iloc[idx, 1])
        plt.show()

    def get_class_distribution(self):
        class_dist = self.img_labels["fire_risk"].value_counts(ascending=True)
        return class_dist

    def get_class_weights(self):
        class_dist = self.get_class_distribution()
        return max(class_dist) / class_dist

    def get_class_idx(self, class_name):
        return self.class2idx[class_name.lower()]

    def get_class_string(self, idx):
        return self.idx2class[idx]

    def get_class_weights_tensor(self):
        class_weights = self.get_class_weights()
        # return tensor as float32
        return torch.tensor(
            [class_weights[self.idx2class[i]] for i in range(len(self.classes))],
            dtype=torch.float32,
        )

    def get_class_from_img_idx(self, idx: int):
        return self.img_labels.iloc[idx, 1]

    def get_class_string_from_numeric_label(self, idxs: list[int]) -> list[str]:
        out = []
        for idx in idxs:
            out.append(self.classes[idx])
        return out


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()