import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Path for IRG images (used to compute NDVI)
IRG_IMG_PATH = "fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"

class CustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir: str,  # Now takes RGB path
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

        # Compute NDVI statistics from IRG images
        if self.ndvi_index:
            self.ndvi_mean, self.ndvi_std = self.__compute_ndvi_stats()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_name = self.img_labels.iloc[idx, 0] + ".png"

        img_path = os.path.join(self.img_dir, img_name)
        rgb_image = Image.open(img_path).convert("RGB")

        if self.task == "classification":
            label = self.class2idx[self.img_labels_dict[self.img_labels.iloc[idx, 0]]]
        else:
            raise ValueError('Task can only be "classification"')

        # Apply transforms before NDVI
        if self.transform:
            rgb_image = self.transform(rgb_image)

        # Compute NDVI from IRG image
        if self.ndvi_index:
            irg_path = os.path.join(IRG_IMG_PATH, img_name)
            irg_image = Image.open(irg_path).convert("RGB")  # IRG images stored as RGB format
            irg_image = self.transform(irg_image)  # Ensure same transformations are applied
            ndvi = self.__compute_ndvi(irg_image)  # Compute NDVI from IRG image
            rgb_image = torch.cat((rgb_image, ndvi.unsqueeze(0)), dim=0)  # Append NDVI as 4th channel

        # Normalize
        if self.normalize_transform:
            rgb_image = self.normalize_transform(rgb_image)

        return rgb_image, label

    def __compute_ndvi(self, irg_image):
        infrared = irg_image[0, :, :]  # IR channel (R in IRG image)
        red = irg_image[2, :, :]  # Red channel (B in IRG image)
        epsilon = 1e-6
        ndvi = (infrared - red) / (infrared + red + epsilon)  # Compute NDVI
        return (ndvi - self.ndvi_mean) / self.ndvi_std  # Normalize NDVI

    def __compute_ndvi_stats(self):
        ndvi_values = []
        for i in range(len(self)):
            img_name = self.img_labels.iloc[i, 0] + ".png"
            irg_path = os.path.join(IRG_IMG_PATH, img_name)
            
            irg_image = Image.open(irg_path).convert("RGB")
            irg_tensor = torch.tensor(np.array(irg_image)).permute(2, 0, 1) / 255.0  # Convert to tensor

            infrared = irg_tensor[0, :, :].numpy().flatten()
            red = irg_tensor[2, :, :].numpy().flatten()
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
