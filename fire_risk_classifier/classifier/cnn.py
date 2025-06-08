import logging

import torch
import torch.nn as nn
from torchvision import models

from fire_risk_classifier.dataclasses.params import Params


def get_cnn_model(params: Params) -> nn.Module:
    algorithm = params.algorithm
    logging.info(f"Using {algorithm} model.")
    match algorithm:
        case "resnet50" | "resnet101" | "resnet152":
            return get_resnet_model(params)
        case "densenet121" | "densenet161" | "densenet169" | "densenet201":
            return get_densenet_model(params)
        case (
            "efficientnet_b4"
            | "efficientnet_b5"
            | "efficientnet_b6"
            | "efficientnet_b7"
        ):
            return get_efficientnet_model(params)
        case "vgg16" | "vgg19" | "vgg19_bn":
            return get_vgg_model(params)
        case _:
            raise ValueError(f"Invalid algorithm: {algorithm}")
    raise ValueError(f"Invalid algorithm: {algorithm}")


def get_classifier_model(params: Params, num_features: int) -> "Classifier":
    return Classifier(
        input_size=num_features,
        num_classes=params.num_labels,
        hidden_size=params.cnn_last_layer_length,
    )


# ------------- ResNet models ------------- #


def get_resnet_model(params: Params) -> models.ResNet:
    algorithm = params.algorithm
    match algorithm:
        case "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        case "resnet101":
            base_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        case "resnet152":
            base_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    num_features = base_model.fc.in_features

    __adapt_model(params, base_model, freeze_layers=False)

    base_model.fc = get_classifier_model(params, num_features)
    return base_model


# ------------- DenseNet models ------------- #


def get_densenet_model(params: Params) -> models.DenseNet:
    algorithm = params.algorithm
    match algorithm:
        case "densenet121":
            base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        case "densenet161":
            base_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        case "densenet169":
            base_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        case "densenet201":
            base_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        case _:
            raise ValueError(f"Invalid algorithm: {algorithm}")

    num_features = base_model.classifier.in_features
    __adapt_model(params, base_model, freeze_layers=False)

    base_model.classifier = get_classifier_model(params, num_features)
    return base_model


# ------------- Other models ------------- #


def get_efficientnet_model(params: Params) -> models.EfficientNet:
    algorithm = params.algorithm
    match algorithm:
        case "efficientnet_b4":
            base_model = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.DEFAULT
            )
        case "efficientnet_b5":
            base_model = models.efficientnet_b5(
                weights=models.EfficientNet_B5_Weights.DEFAULT
            )
        case "efficientnet_b6":
            base_model = models.efficientnet_b6(
                weights=models.EfficientNet_B6_Weights.DEFAULT
            )
        case "efficientnet_b7":
            base_model = models.efficientnet_b7(
                weights=models.EfficientNet_B7_Weights.DEFAULT
            )

    num_features = base_model.classifier[1].in_features
    __adapt_model(params, base_model, freeze_layers=False)

    if isinstance(base_model.classifier, nn.Linear):
        base_model.classifier = get_classifier_model(params, num_features)
    else:
        base_model.classifier[1] = get_classifier_model(params, num_features)

    return base_model


def get_vgg_model(params: Params) -> models.VGG:
    algorithm = params.algorithm
    match algorithm:
        case "vgg16":
            base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        case "vgg19":
            base_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        case "vgg19_bn":
            base_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

    num_features = base_model.classifier[6].in_features
    __adapt_model(params, base_model, freeze_layers=False)

    base_model.classifier[6] = get_classifier_model(params, num_features)
    return base_model


def __adapt_model(
    params: Params,
    base_model: nn.Module,
    freeze_layers: bool = True
):
    if params.calculate_ndvi_index:
        __adapt_model_to_ndvi(base_model)
    elif params.calculate_rgbi:
        __adapt_model_to_rgbi(base_model)

    print(params.__dict__)

    for param in base_model.parameters():
        param.requires_grad = not freeze_layers

    # Ensure BatchNorm layers are always trainable
    for module in base_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True



def __adapt_model_to_ndvi(base_model: nn.Module):
    """
    Modify the first convolutional layer of the CNN to accept 4 input channels instead of 3 (RGB + NDVI).
    """
    first_layer = None

    # Identify the first layer based on model type
    if isinstance(base_model, models.ResNet):
        first_layer = base_model.conv1
    elif isinstance(base_model, models.DenseNet):
        first_layer = base_model.features.conv0
    elif isinstance(base_model, models.EfficientNet):
        first_layer = base_model.features[0][0]  # EfficientNet's Conv2d is inside Conv2dNormActivation

    if first_layer is None:
        raise ValueError("Unsupported model type for 4 channel adaptation.")

    # Check if bias is used
    bias = first_layer.bias is not None if hasattr(first_layer, "bias") else False

    # Create a new conv layer with 4 input channels
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=bias,
    )

    with torch.no_grad():
        # Maintain the original RGB mapping
        new_conv.weight[:, 0, :, :].copy_(first_layer.weight[:, 0, :, :])  # R → R
        new_conv.weight[:, 1, :, :].copy_(first_layer.weight[:, 1, :, :])  # G → G
        new_conv.weight[:, 2, :, :].copy_(first_layer.weight[:, 2, :, :])  # B → B

        # NDVI: Initialize based on Red (or use a different strategy if needed)
        new_conv.weight[:, 3, :, :].copy_(first_layer.weight[:, 0, :, :])  # Copy Red weights for NDVI

    # Copy bias if applicable
    if bias:
        new_conv.bias.data.copy_(first_layer.bias.data)

    # Replace the old conv layer
    if isinstance(base_model, models.ResNet):
        base_model.conv1 = new_conv
    elif isinstance(base_model, models.DenseNet):
        base_model.features.conv0 = new_conv
    elif isinstance(base_model, models.EfficientNet):
        base_model.features[0][0] = new_conv  # EfficientNet's conv layer is inside a wrapper

    return base_model


def __adapt_model_to_rgbi(base_model: nn.Module):
    """
    Modify the first conv layer to accept 4 input channels (RGB + Infrared).
    """
    first_layer = None

    if isinstance(base_model, models.ResNet):
        first_layer = base_model.conv1
    elif isinstance(base_model, models.DenseNet):
        first_layer = base_model.features.conv0
    elif isinstance(base_model, models.EfficientNet):
        first_layer = base_model.features[0][0]
    else:
        raise ValueError("Unsupported model type for RGBI adaptation.")

    bias = first_layer.bias is not None if hasattr(first_layer, "bias") else False

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=bias,
    )

    with torch.no_grad():
        # Copy original weights
        new_conv.weight[:, :3, :, :] = first_layer.weight[:, :3, :, :]

        # Initialize IR weights
        new_conv.weight[:, 3, :, :] = first_layer.weight[:, 0, :, :]  # Copy Red weights for IR init (or use zeros)

    if bias:
        new_conv.bias.data.copy_(first_layer.bias.data)

    # Replace the original conv layer
    if isinstance(base_model, models.ResNet):
        base_model.conv1 = new_conv
    elif isinstance(base_model, models.DenseNet):
        base_model.features.conv0 = new_conv
    elif isinstance(base_model, models.EfficientNet):
        base_model.features[0][0] = new_conv

    return base_model


class Classifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Classifier, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        n_classes = 1 if num_classes == 2 else num_classes
        self.fc3 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x
