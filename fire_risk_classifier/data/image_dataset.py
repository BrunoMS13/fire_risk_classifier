import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
        self.img_labels_dict = {row[0]: row[1] for row in self.img_labels.itertuples(index=False)}

        classes = sorted(self.img_labels.iloc[:, 1].unique())
        self.img_dir = img_dir
        self.transform = transform
        self.normalize_transform = normalize_transform
        self.task = task.lower()
        self.ndvi_index = ndvi_index

        self.classes = list(classes)
        self.class2idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx2class = {i: self.classes[i] for i in range(len(self.classes))}

        # Compute NDVI statistics
        self.ndvi_mean, self.ndvi_std = self.__compute_ndvi_stats()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = Image.open(img_path).convert("RGB")

        if self.task == "classification":
            label = self.class2idx[self.img_labels_dict[self.img_labels.iloc[idx, 0]]]
        else:
            raise ValueError('Task can only be "classification"')

        # Apply transforms before NDVI
        if self.transform:
            image = self.transform(image)

        # Compute NDVI before normalizing
        if self.ndvi_index:
            ndvi = self.__compute_ndvi(image)
            image = torch.cat((image, ndvi.unsqueeze(0)), dim=0)

        # Normalize
        if self.normalize_transform:
            image = self.normalize_transform(image)

        return image, label

    def __compute_ndvi(self, image):
        infrared = image[0, :, :]
        red = image[2, :, :]
        epsilon = 1e-6
        ndvi = (infrared - red) / (infrared + red + epsilon)
        return (ndvi - self.ndvi_mean) / self.ndvi_std

    def __compute_ndvi_stats(self):
        ndvi_values = []
        for i in range(len(self)):
            img, _ = self[i]  # Get image tensor
            infrared = img[0, :, :].numpy().flatten()
            red = img[2, :, :].numpy().flatten()
            ndvi = (infrared - red) / (infrared + red + 1e-6)
            ndvi_values.extend(ndvi)

        return np.mean(ndvi_values), np.std(ndvi_values)

    def get_class_distribution(self):
        return self.img_labels["fire_risk"].value_counts(ascending=True)

    def get_class_weights(self):
        class_dist = self.get_class_distribution()
        return max(class_dist) / class_dist

    def get_class_weights_tensor(self):
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[self.idx2class[i]] for i in range(len(self.classes))], dtype=torch.float32)
