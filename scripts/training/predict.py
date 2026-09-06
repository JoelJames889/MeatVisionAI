import torch
from PIL import Image
import sys
import os

from model import MeatVisionModel
from utils import get_test_transform

# The exact classes our models learned
SPECIES_CLASSES = ["beef", "chicken", "fish", "pork"]
FRESHNESS_CLASSES = ["fresh", "halffresh", "spoiled"]

# Model Paths
SPECIES_MODEL_PATH = r"d:\MeatVision_Project\Models\species_model.pth"
FRESHNESS_MODEL_PATH = r"d:\MeatVision_Project\Models\freshness_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    # Load Species Model (4 classes)
    species_model = MeatVisionModel(num_classes=4).get()
    species_model.load_state_dict(
        torch.load(SPECIES_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    species_model = species_model.to(DEVICE)
    species_model.eval()

    # Load Freshness Model (3 classes)
    freshness_model = MeatVisionModel(num_classes=3).get()
    freshness_model.load_state_dict(
        torch.load(FRESHNESS_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    freshness_model = freshness_model.to(DEVICE)
    freshness_model.eval()

    return species_model, freshness_model


def predict(image_path):
    print(f"\n[Analyzing Meat: {os.path.basename(image_path)}...]")

    species_model, freshness_model = load_models()
    transform = get_test_transform()

    try:
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Predict Species
        with torch.no_grad():
            species_outputs = species_model(image_tensor)
            species_probs = torch.nn.functional.softmax(species_outputs, dim=1)
            species_conf, species_idx = torch.max(species_probs, 1)
            species_pred = SPECIES_CLASSES[species_idx.item()]

            # Predict Freshness
            freshness_outputs = freshness_model(image_tensor)
            freshness_probs = torch.nn.functional.softmax(freshness_outputs, dim=1)
            freshness_conf, freshness_idx = torch.max(freshness_probs, 1)
            freshness_pred = FRESHNESS_CLASSES[freshness_idx.item()]

        print("\n" + "=" * 40)
        print("MEATVISION RESULTS")
        print("=" * 40)
        print(
            f"Species   : {species_pred.upper()} ({species_conf.item()*100:.1f}% Confidence)"
        )
        print(
            f"Freshness : {freshness_pred.upper()} ({freshness_conf.item()*100:.1f}% Confidence)"
        )
        print("=" * 40 + "\n")

        return {
            "species": species_pred,
            "freshness": freshness_pred,
        }
    except Exception as e:
        print(f"[Error analyzing image: {e}]")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
