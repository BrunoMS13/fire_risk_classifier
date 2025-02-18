import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


IRG_IMG_PATH = "fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"


class CustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        annotations_file: str,
        transform=transforms.ToTensor(),
        ndvi_index: bool = False,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels_dict = {row[0]: row[1] for row in self.img_labels.itertuples(index=False)}

        classes = sorted(self.img_labels.iloc[:, 1].unique())
        self.img_dir = img_dir
        self.transform = transform
        self.ndvi_index = ndvi_index
        self.normalize_transform = self.__get_normalize_transform()

        self.classes = list(classes)
        self.class2idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx2class = {i: self.classes[i] for i in range(len(self.classes))}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_name = self.img_labels.iloc[idx, 0] + ".png"
        img_path = os.path.join(self.img_dir, img_name)

        main_image = Image.open(img_path).convert("RGB")
        label = self.class2idx[self.img_labels_dict[self.img_labels.iloc[idx, 0]]]
        
        # Data augmentation techniques
        main_image = self.transform(main_image)

        if self.ndvi_index:
            main_image = self.__add_ndvi_as_fourth_channel(main_image, img_name)

        # Compute data normalization
        main_image = self.normalize_transform(main_image)
        return main_image, label

    def __add_ndvi_as_fourth_channel(self, main_image, img_name):
        """Compute NDVI and append it as the 4th channel to the main image."""
        irg_path = os.path.join(IRG_IMG_PATH, img_name)
        irg_image = Image.open(irg_path).convert("RGB")  # IRG images stored as RGB format

        # Apply the same transformations to IRG image
        irg_image = self.transform(irg_image)

        # Compute NDVI without extra normalization
        ndvi = self.__compute_ndvi(irg_image)

        # Append NDVI as the fourth channel
        return torch.cat((main_image, ndvi.unsqueeze(0)), dim=0)

    def __compute_ndvi(self, irg_image):
        infrared = irg_image[0, :, :]
        red = irg_image[2, :, :]
        return (infrared - red) / (infrared + red + 1e-6)

    def get_class_distribution(self):
        return self.img_labels["fire_risk"].value_counts(ascending=True)

    def get_class_weights(self):
        class_dist = self.get_class_distribution()
        return max(class_dist) / class_dist

    def get_class_weights_tensor(self):
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[self.idx2class[i]] for i in range(len(self.classes))], dtype=torch.float32)

    def __get_normalize_transform(self) -> transforms.Normalize:
        if "RGB" in self.img_dir:
            dataset_mean = RGB_DATA["mean"]
            dataset_std = RGB_DATA["std"]
            if self.ndvi_index:
                dataset_mean = dataset_mean + NDVI_DATA["mean"]
                dataset_std = dataset_std + NDVI_DATA["std"]
        elif "IRG" in self.img_dir:
            dataset_mean = IRG_DATA["mean"]
            dataset_std = IRG_DATA["std"]
        return transforms.Normalize(mean=dataset_mean, std=dataset_std)

# Calculated mean and std for IRG, RGB images and NDVI 
IRG_DATA = {"mean": [0.5129562020301819, 0.4104961156845093, 0.3838688135147095], "std": [0.1694248914718628, 0.19696880877017975, 0.16279979050159454]} # mean, std
RGB_DATA = {"mean": [0.41085049510002136, 0.3847522735595703, 0.34787386655807495], "std": [0.19633813202381134, 0.1622946709394455, 0.13759763538837433]} # mean, std
NDVI_DATA = {"mean": [0.15421810746192932], "std": [0.19456863403320312]} # mean, std