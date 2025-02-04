import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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
