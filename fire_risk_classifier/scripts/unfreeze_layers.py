import torch.nn as nn
from fire_risk_classifier.dataclasses.params import Params

class UnfreezeLayers:
    @classmethod
    def unfreeze_layers(cls, model: nn.Module, current_epoch: int, params: Params):
        """Gradually unfreeze layers based on CNN architecture."""
        if params.algorithm.startswith("efficientnet"):
            cls.__efficient_net_gradual_unfreeze(model, current_epoch)
        elif params.algorithm.startswith("resnet"):
            cls.__resnet_gradual_unfreeze(model, current_epoch)
        elif params.algorithm.startswith("densenet"):
            cls.__densenet_gradual_unfreeze(model, current_epoch)

    @classmethod
    def __efficient_net_gradual_unfreeze(cls, model: nn.Module, current_epoch: int):
        """Gradually unfreeze EfficientNet layers at different epochs."""
        layer_groups = [
            model.features[:3],  # Early feature extraction layers
            model.features[3:6],  # Mid-level layers
            model.features[6:],  # Deeper high-level layers
        ]

        # Freeze all layers initially
        for group in layer_groups:
            for param in group.parameters():
                param.requires_grad = False

        # Always keep classifier head trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Gradually unfreeze at different epochs
        if current_epoch >= 2:
            for param in layer_groups[0].parameters():
                param.requires_grad = True

        if current_epoch >= 4:
            for param in layer_groups[1].parameters():
                param.requires_grad = True

        if current_epoch >= 6:
            for param in layer_groups[2].parameters():
                param.requires_grad = True

    @classmethod
    def __resnet_gradual_unfreeze(cls, model: nn.Module, current_epoch: int):
        """Gradually unfreeze ResNet layers (ResNet50, ResNet101, etc.)."""
        layer_groups = [
            model.conv1,  # First conv layer
            model.layer1,  # Initial residual blocks
            model.layer2,  # Middle residual blocks
            model.layer3,  # Deeper residual blocks
            model.layer4,  # Final residual blocks
        ]

        # Freeze all layers initially
        for group in layer_groups:
            for param in group.parameters():
                param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

        # Unfreeze gradually
        if current_epoch >= 2:
            for param in layer_groups[1].parameters():
                param.requires_grad = True  # Unfreeze layer1

        if current_epoch >= 4:
            for param in layer_groups[2].parameters():
                param.requires_grad = True  # Unfreeze layer2

        if current_epoch >= 6:
            for param in layer_groups[3].parameters():
                param.requires_grad = True  # Unfreeze layer3

        if current_epoch >= 8:
            for param in layer_groups[4].parameters():
                param.requires_grad = True  # Unfreeze layer4

    @classmethod
    def __densenet_gradual_unfreeze(cls, model: nn.Module, current_epoch: int):
        """Gradually unfreeze DenseNet layers."""
        layer_groups = [
            model.features[:6],  # Initial layers
            model.features[6:12],  # Middle layers
            model.features[12:],  # Deeper layers
        ]

        # Freeze all layers initially
        for group in layer_groups:
            for param in group.parameters():
                param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

        # Gradually unfreeze layers
        if current_epoch >= 2:
            for param in layer_groups[0].parameters():
                param.requires_grad = True  # Unfreeze first layers

        if current_epoch >= 4:
            for param in layer_groups[1].parameters():
                param.requires_grad = True  # Unfreeze middle layers

        if current_epoch >= 6:
            for param in layer_groups[2].parameters():
                param.requires_grad = True  # Unfreeze final layers
