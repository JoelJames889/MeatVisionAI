import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


class MeatVisionModel:

    def __init__(self, num_classes: int, backbone: str = "efficientnet_b0"):
        self.backbone = backbone.lower()
        if self.backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.model = convnext_tiny(weights=weights)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, num_classes)
        else:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.model = efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def get(self):
        return self.model
