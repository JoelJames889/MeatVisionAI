import os
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

species_model = MeatVisionModel(len(SPECIES_CLASSES)).get().to(DEVICE)
freshness_model = MeatVisionModel(len(FRESHNESS_CLASSES)).get().to(DEVICE)


def reload_weights():
    try:
        if os.path.exists(SPECIES_MODEL):
            species_model.load_state_dict(torch.load(SPECIES_MODEL, map_location=DEVICE))
            species_model.eval()
        if os.path.exists(FRESHNESS_MODEL):
            freshness_model.load_state_dict(torch.load(FRESHNESS_MODEL, map_location=DEVICE))
            freshness_model.eval()
    except Exception:
        pass


reload_weights()
print("AI Models loaded successfully. System ready.")


# ==========================================================
# STATUS MAPS & INSPECTION GUIDELINES
# ==========================================================

FRESHNESS_STATUS_MAP = {
    "Fresh": {
        "status": "Grade A - Peak Freshness",
        "description": "Safe for consumption. Preserved texture, natural pigmentation, and no microbial spoilage detected.",
        "badge_class": "badge-fresh",
        "color": "#10b981",  # Emerald green
    },
    "Half Fresh": {
        "status": "Grade B - Marginal / Cook Immediately",
        "description": "Signs of early biochemical oxidation detected. Suitable only if cooked thoroughly and consumed promptly.",
        "badge_class": "badge-half-fresh",
        "color": "#f59e0b",  # Amber
    },
    "Spoiled": {
        "status": "Grade C - Spoilage Confirmed",
        "description": "Significant decomposition, surface discoloration, or severe chemical breakdown. Unfit for consumption.",
        "badge_class": "badge-spoiled",
        "color": "#ef4444",  # Red
    },
}

SPECIES_INSPECTION_GUIDANCE = {
    "Beef": "Inspection evaluates oxymyoglobin/metmyoglobin surface oxidation, fat marbling tint, and tissue elasticity.",
    "Chicken": "Inspection evaluates muscle translucency, absence of yellowed/greyish exudate, and surface sheen.",
    "Fish": "Inspection evaluates muscle opacity, dermal color vibrancy, firmness, and absence of milky surface film.",
    "Pork": "Inspection evaluates pale-pink pigment consistency, fat firmness, and lack of dark-grey/greenish discoloration.",
}


# ==========================================================
# PREDICTION LOGIC
# ==========================================================

def predict(image_path: str) -> dict:
    reload_weights()
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

    species_name = SPECIES_CLASSES[species_index.item()]
    freshness_grade = FRESHNESS_CLASSES[freshness_index.item()]

    # Extract full probability distribution
    species_prob_dist = {
        cls_name: round(prob.item() * 100, 2)
        for cls_name, prob in zip(SPECIES_CLASSES, species_prob[0])
    }
    freshness_prob_dist = {
        cls_name: round(prob.item() * 100, 2)
        for cls_name, prob in zip(FRESHNESS_CLASSES, freshness_prob[0])
    }

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
            "species_probabilities": species_prob_dist,
            "freshness": "Invalid Image",
            "freshness_confidence": freshness_conf_pct,
            "freshness_probabilities": freshness_prob_dist,
            "freshness_status": "Rejection - Non-Meat Upload",
            "freshness_description": "Uploaded input was flagged as out-of-distribution or synthetic.",
            "freshness_color": "#ef4444",
            "quality_guidance": "Please upload a clear, high-resolution photo of raw meat or fish.",
        }

    freshness_info = FRESHNESS_STATUS_MAP.get(freshness_grade, {})

    return {
        "is_valid": True,
        "species": species_name,
        "species_confidence": species_conf_pct,
        "species_probabilities": species_prob_dist,
        "freshness": freshness_grade,
        "freshness_confidence": freshness_conf_pct,
        "freshness_probabilities": freshness_prob_dist,
        "freshness_status": freshness_info.get("status", freshness_grade),
        "freshness_description": freshness_info.get("description", ""),
        "freshness_color": freshness_info.get("color", "#10b981"),
        "quality_guidance": SPECIES_INSPECTION_GUIDANCE.get(
            species_name, "Standard optical inspection applied."
        ),
    }

