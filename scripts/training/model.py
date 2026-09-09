import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class MeatVisionModel:

    def __init__(self, num_classes):
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.model = convnext_tiny(weights=weights)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def get(self):
        return self.model
