import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        annotations_file: str,
        transform=None,
        normalize_transform=None,
        ndvi_index: bool = False,
        task: str = "classification",
    ):

        self.img_labels = pd.read_csv(annotations_file)
        classes = sorted(self.img_labels.iloc[:, 1].unique())

        self.img_dir = img_dir
        self.transform = transform
        self.normalize_transform = normalize_transform

        self.task = task.lower()
        self.ndvi_index = ndvi_index

        self.classes = list(classes)
        self.class2idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx2class = {i: self.classes[i] for i in range(len(self.classes))}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.task == "classification":
            label = self.class2idx[self.img_labels.iloc[idx, 1]]
        else:
            raise ValueError('Task can only be "classification"')

        if self.transform:
            image = self.transform(image)

        if self.ndvi_index:
            image = self.__get_image_with_ndvi_index(image)

        image = self.normalize_transform(image)

        return image, label

    def __get_image_with_ndvi_index(self, image):
        infrared = image[0, :, :]
        red = image[2, :, :]

        # NDVI = (NIR - RED) / (NIR + RED) adding epsilon to avoid division by zero.
        epsilon = 1e-6
        ndvi = (infrared - red) / (infrared + red + epsilon)
        return torch.cat((image, ndvi.unsqueeze(0)), dim=0)

    def show(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = read_image(img_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")
        plt.title(self.img_labels.iloc[idx, 1])
        plt.show()

    def get_class_distribution(self):
        return self.img_labels["fire_risk"].value_counts(ascending=True)

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
        return [self.classes[idx] for idx in idxs]


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
