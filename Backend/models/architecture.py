import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MeatVisionModel:
    """
    Deep Learning Model Architecture for MeatVision AI.
    Uses Transfer Learning with EfficientNet-B0 pretrained backbone.
    """

    def __init__(self, num_classes: int):
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def get(self) -> nn.Module:
        return self.model
