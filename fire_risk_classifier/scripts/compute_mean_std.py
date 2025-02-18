import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from fire_risk_classifier.data.image_dataset import CustomImageDataset


# Path for IRG images (used to compute NDVI)
IRG_IMG_PATH = "fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"
RGB_IMG_PATH = "fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed"


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size
        num_pixels += batch_samples * images.size(2) * images.size(3)  # Total pixel count (H * W * batch)

        # Compute sum across all pixels
        mean += images.sum(dim=[0, 2, 3])  # Sum over batch, height, width
        std += (images ** 2).sum(dim=[0, 2, 3])  # Sum of squared pixel values

    # Compute final mean and std
    mean /= num_pixels
    std = torch.sqrt(std / num_pixels - mean ** 2)  # Variance formula: E[X^2] - (E[X])^2

    return mean.tolist(), std.tolist()

def compute_ndvi_mean_std(self):
    ndvi_values = []
    for i in range(len(self)):
        img_name = f"{self.img_labels.iloc[i, 0]}.png"
        irg_path = os.path.join(IRG_IMG_PATH, img_name)

        irg_image = Image.open(irg_path).convert("RGB")
        irg_tensor = torch.tensor(np.array(irg_image)).permute(2, 0, 1) / 255.0  # Convert to tensor

        infrared = irg_tensor[0, :, :].numpy().flatten()
        red = irg_tensor[2, :, :].numpy().flatten()
        ndvi = (infrared - red) / (infrared + red + 1e-6)
        ndvi_values.extend(ndvi)

    return np.mean(ndvi_values), np.std(ndvi_values)



def main():
    transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ColorJitter(brightness=0.2),
                    transforms.ToTensor(),
                ]
            )

    img_dir = "fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"
    annotations_file = "fire_risk_classifier/data/csvs/train_2classes.csv"

    # Training data loader
    dataset = CustomImageDataset(
        img_dir,
        annotations_file,
        transform,
        False
    )
    print(compute_mean_std(dataset))


IRG = [0.5129562020301819, 0.4104961156845093, 0.3838688135147095], [0.1694248914718628, 0.19696880877017975, 0.16279979050159454] # mean, std
RGB = [0.41085049510002136, 0.3847522735595703, 0.34787386655807495], [0.19633813202381134, 0.1622946709394455, 0.13759763538837433] # mean, std
NDVI = [0.15421810746192932, 0.19456863403320312] # mean, std