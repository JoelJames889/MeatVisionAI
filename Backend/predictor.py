import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from backend.models.architecture import MeatVisionModel
from backend.preprocess import preprocess
from backend.utils import load_image
from backend.config import SPECIES_MODEL, FRESHNESS_MODEL
from backend.validator import validate_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# CLASS NAMES
# ==========================================================

SPECIES_CLASSES = ["Beef", "Chicken", "Fish", "Pork"]
FRESHNESS_CLASSES = ["Fresh", "Half Fresh", "Spoiled"]

# ==========================================================
# LOAD MODELS
# ==========================================================

print(f"Loading AI Models on device: {DEVICE}...")

species_model = MeatVisionModel(len(SPECIES_CLASSES)).get()
species_model.load_state_dict(torch.load(SPECIES_MODEL, map_location=DEVICE))
species_model.eval()
species_model.to(DEVICE)

freshness_model = MeatVisionModel(len(FRESHNESS_CLASSES)).get()
freshness_model.load_state_dict(torch.load(FRESHNESS_MODEL, map_location=DEVICE))
freshness_model.eval()
freshness_model.to(DEVICE)

print("AI Models loaded successfully. System ready.")


# ==========================================================
# PREDICTION LOGIC
# ==========================================================

def predict(image_path: str) -> dict:
    image = load_image(image_path)
    tensor = preprocess(image).to(DEVICE)

    with torch.no_grad():
        # Species Prediction
        species_output = species_model(tensor)
        species_prob = F.softmax(species_output, dim=1)
        species_confidence, species_index = torch.max(species_prob, dim=1)

        # Freshness Prediction
        freshness_output = freshness_model(tensor)
        freshness_prob = F.softmax(freshness_output, dim=1)
        freshness_confidence, freshness_index = torch.max(freshness_prob, dim=1)

    species_conf_pct = round(species_confidence.item() * 100, 2)
    freshness_conf_pct = round(freshness_confidence.item() * 100, 2)

    # Validate whether the image is meat/food or an invalid non-meat image
    val_res = validate_image(image_path, species_conf_pct)

    if not val_res["is_valid"]:
        return {
            "is_valid": False,
            "error_message": val_res["message"],
            "error_code": val_res["error_code"],
            "detected_category": val_res["detected_category"],
            "species": "Invalid Image",
            "species_confidence": species_conf_pct,
            "freshness": "Invalid Image",
            "freshness_confidence": freshness_conf_pct,
        }

    return {
        "is_valid": True,
        "species": SPECIES_CLASSES[species_index.item()],
        "species_confidence": species_conf_pct,
        "freshness": FRESHNESS_CLASSES[freshness_index.item()],
        "freshness_confidence": freshness_conf_pct,
    }

