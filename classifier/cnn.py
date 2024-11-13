import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from fire_risk_classifier.dataclasses.params import Params


def get_cnn_model(params: Params) -> nn.Module:
    print(f"Using {params.algorithm} model.")
    if params.algorithm == "resnet":
        return get_resnet_model(params)
    if params.algorithm == "densenet":
        return get_densenet_model(params)
    raise ValueError(f"Invalid algorithm: {params.algorithm}")


def get_classifier_model(params: Params, num_features: int) -> "Classifier":
    classifier = Classifier(
        input_size=num_features,
        num_classes=params.num_labels,
        hidden_size=params.cnn_last_layer_length,
    )
    return classifier


def get_resnet_model(params: Params):
    logging.info("Using ResNet50 model.")
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = base_model.fc.in_features

    for param in base_model.parameters():
        # Set to False if you want to freeze layers of feature extractor.
        param.requires_grad = True

    base_model.fc = get_classifier_model(params, num_features)
    return base_model


def get_densenet_model(params: Params):
    logging.info("Using DenseNet161 model.")
    base_model = models.densenet161(pretrained=True)
    num_features = base_model.classifier.in_features

    for param in base_model.parameters():
        # Set to False if you want to freeze layers of feature extractor.
        param.requires_grad = True

    base_model.classifier = get_classifier_model(params, num_features)
    return base_model


class Classifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Classifier, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x
