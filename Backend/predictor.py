import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

# ==========================================================
# Add Scripts/models to Python Path
# ==========================================================

MODELS_SCRIPT = PROJECT_ROOT / "Scripts" / "models"
if str(MODELS_SCRIPT) not in sys.path:
    sys.path.append(str(MODELS_SCRIPT))

# ==========================================================
# Import Existing Model
# ==========================================================

from model import MeatVisionModel

from Backend.preprocess import preprocess
from Backend.utils import load_image
from Backend.config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# CLASS NAMES
# ==========================================================

SPECIES_CLASSES = ["Beef", "Chicken", "Fish", "Pork"]

FRESHNESS_CLASSES = ["Fresh", "Half Fresh", "Spoiled"]

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading AI Models...")

species_model = MeatVisionModel(4).get()

species_model.load_state_dict(torch.load(SPECIES_MODEL, map_location=DEVICE))

species_model.eval()
species_model.to(DEVICE)

print("Species Model Loaded")

freshness_model = MeatVisionModel(3).get()

freshness_model.load_state_dict(torch.load(FRESHNESS_MODEL, map_location=DEVICE))

freshness_model.eval()
freshness_model.to(DEVICE)

print("Freshness Model Loaded")

print("System Ready")

# ==========================================================
# Prediction Function
# ==========================================================


def predict(image_path: str) -> dict:

    image = load_image(image_path)

    tensor = preprocess(image)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        # Species Prediction
        species_output = species_model(tensor)

        species_prob = F.softmax(species_output, dim=1)

        species_confidence, species_index = torch.max(species_prob, dim=1)

        # Freshness Prediction
        freshness_output = freshness_model(tensor)

        freshness_prob = F.softmax(freshness_output, dim=1)

        freshness_confidence, freshness_index = torch.max(freshness_prob, dim=1)

    return {
        "species": SPECIES_CLASSES[species_index.item()],
        "species_confidence": round(species_confidence.item() * 100, 2),
        "freshness": FRESHNESS_CLASSES[freshness_index.item()],
        "freshness_confidence": round(freshness_confidence.item() * 100, 2),
    }
