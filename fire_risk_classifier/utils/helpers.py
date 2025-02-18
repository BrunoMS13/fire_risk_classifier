import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from fire_risk_classifier.data.image_dataset import CustomImageDataset


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size
        images = images.view(batch_samples, images.size(1), -1)  # Flatten HxW
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    return mean.tolist(), std.tolist()


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
dataset = CustomImageDataset(img_dir, annotations_file, transform, False)
