import logging

import torch
import torch.nn as nn
from torchvision import models

from fire_risk_classifier.dataclasses.params import Params


# t
def get_cnn_model(params: Params) -> nn.Module:
    if params.algorithm == "resnet":
        return get_resnet_model(params)
    if params.algorithm == "densenet":
        return get_densenet_model(params)
    if params.algorithm == "inception":
        return get_inception_model(params)
    if params.algorithm == "efficientnet":
        return get_efficientnet_model(params)
    if params.algorithm == "vgg":
        return get_vgg_model(params)
    raise ValueError(f"Invalid algorithm: {params.algorithm}")


def get_classifier_model(params: Params, num_features: int) -> "Classifier":
    return Classifier(
        input_size=num_features,
        num_classes=params.num_labels,
        hidden_size=params.cnn_last_layer_length,
    )


def get_resnet_model(params: Params) -> models.ResNet:
    logging.info("Using ResNet50 model.")
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = base_model.fc.in_features

    __adapt_model(params.calculate_ndvi_index, base_model, freeze_layers=True)

    base_model.fc = get_classifier_model(params, num_features)
    return base_model


def get_densenet_model(params: Params) -> models.DenseNet:
    logging.info("Using DenseNet161 model.")
    base_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    num_features = base_model.classifier.in_features

    __adapt_model(params.calculate_ndvi_index, base_model, freeze_layers=True)

    base_model.classifier = get_classifier_model(params, num_features)
    return base_model


def get_inception_model(params: Params) -> models.Inception3:
    logging.info("Using InceptionV3 model.")
    base_model = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True
    )
    num_features = base_model.fc.in_features

    __adapt_model(params.calculate_ndvi_index, base_model, freeze_layers=True)

    base_model.fc = get_classifier_model(params, num_features)
    return base_model


def get_efficientnet_model(params: Params) -> models.EfficientNet:
    logging.info("Using EfficientNetB4 model.")
    base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    num_features = base_model.classifier[1].in_features

    __adapt_model(params.calculate_ndvi_index, base_model, freeze_layers=True)

    base_model.classifier[1] = get_classifier_model(params, num_features)
    return base_model


def get_vgg_model(params: Params) -> models.VGG:
    logging.info("Using VGG16 model.")
    base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    num_features = base_model.classifier[6].in_features

    __adapt_model(params.calculate_ndvi_index, base_model, freeze_layers=True)

    base_model.classifier[6] = get_classifier_model(params, num_features)
    return base_model


def __adapt_model(
    calculate_ndvi_index: bool, base_model: nn.Module, freeze_layers: bool
):
    if calculate_ndvi_index:
        __adapt_model_to_ndvi(base_model)

    for param in base_model.parameters():
        param.requires_grad = not freeze_layers


def __adapt_model_to_ndvi(self, base_model: nn.Module):
    original_conv0 = base_model.features.conv0
    base_model.features.conv0 = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv0.out_channels,
        kernel_size=original_conv0.kernel_size,
        stride=original_conv0.stride,
        padding=original_conv0.padding,
        bias=False,
    )
    # Initialize weights for the new channel
    with torch.no_grad():
        # Copy RGB weights
        base_model.features.conv0.weight[:, :3, :, :] = original_conv0.weight
        # Copy NIR weights
        base_model.features.conv0.weight[:, 3, :, :] = original_conv0.weight[:, 0, :, :]


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
