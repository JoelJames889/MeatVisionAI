import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MeatVisionModel:

    def __init__(self, num_classes):
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def get(self):
        return self.model
