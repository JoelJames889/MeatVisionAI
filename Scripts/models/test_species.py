import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image

from config import *

# -----------------------------
# Classes
# -----------------------------

classes = ["beef", "chicken", "fish", "pork"]

# -----------------------------
# Model
# -----------------------------

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(in_features, len(classes))

model.load_state_dict(torch.load(MODEL_DIR / "species_model.pth", map_location=DEVICE))

model.to(DEVICE)

model.eval()

# -----------------------------
# Transform
# -----------------------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# -----------------------------
# Image Path
# -----------------------------

IMAGE_PATH = (
    r"D:\MeatVision_Project\Dataset\species\beef\04fd788d92224ba2bac15e532af9d86d.jpg"
)

# Put image path above

image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(DEVICE)

# -----------------------------
# Prediction
# -----------------------------

with torch.no_grad():

    output = model(image)

    _, pred = torch.max(output, 1)

print()

print("=" * 50)

print("Prediction :", classes[pred.item()])

print("=" * 50)
