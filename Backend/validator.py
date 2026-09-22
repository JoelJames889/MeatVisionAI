import re
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = MobileNet_V3_Small_Weights.DEFAULT
imagenet_model = models.mobilenet_v3_small(weights=weights).eval().to(DEVICE)
categories = weights.meta["categories"]
transform = weights.transforms()

# Explicit non-food object categories to reject
EXPLICIT_NON_FOOD_KEYWORDS = [
    # Documents, UI, Digital Media & Office Items
    "web site", "envelope", "menu", "book jacket", "comic book", "binder", "notebook",
    "screen", "monitor", "television", "tv", "scoreboard", "billboard", "street sign", "poster",
    "paper towel", "toilet tissue", "rubber eraser", "ruler", "pencil box", "pencil sharpener",
    "ballpoint", "fountain pen", "quill", "packet", "carton", "digital clock", "cash machine", "slot",
    "safe", "matchstick", "lighter", "candle", "vase", "pottery", "pillow", "quilt", "curtain", "doormat",
    # Vehicles & Transportation
    "car", "sports car", "convertible", "limousine", "minivan", "pickup", "cab", "racer",
    "automobile", "truck", "bus", "airplane", "aircraft", "bicycle", "motorcycle",
    "train", "boat", "ship", "submarines", "tractor", "trailer",
    # Electronics & Appliances
    "laptop", "computer", "keyboard", "mouse", "telephone", "phone", "cellular", "camera",
    "radio", "speaker", "headphone", "modem", "refrigerator", "microwave", "clock",
    # Furniture & Household
    "sofa", "couch", "chair", "desk", "lamp", "bed", "wardrobe", "bookcase", "bathtub",
    # Apparel & Personal Items
    "shoe", "boot", "sneaker", "sandal", "glove", "hat", "cap", "jacket", "coat",
    "shirt", "trousers", "pants", "dress", "skirt", "sock", "tie", "umbrella",
    "sunglasses", "backpack", "handbag", "wallet",
    # Animals (Non-Food Pets / Wildlife)
    "cat", "dog", "puppy", "kitten", "horse", "lion", "tiger", "bear", "elephant",
    "giraffe", "monkey", "penguin", "zebra", "cheetah", "leopard", "fox", "wolf",
    # Architecture & Scenery
    "building", "house", "church", "bridge", "tower", "castle", "skyscraper", "traffic light",
    "mountain", "alp", "volcano", "lakeside", "promontory"
]

# Exceptions / Food contexts that shouldn't trigger rejection
ALLOWED_FOOD_CONTEXTS = [
    "butcher shop", "plate", "tray", "cleaver", "frying pan", "wok", "platter", "bowl",
    "plastic bag", "ice cream", "trifle", "conch", "pomegranate", "head cabbage", "cauliflower",
    "squash", "hot pot", "dough", "carbonara", "meat loaf", "cheeseburger", "hotdog",
    "rotisserie", "nematode", "gar", "barracouta", "coho", "puffer", "tench", "goldfish",
    "shower cap", "velvet", "wool", "hog", "sea snake", "sea cucumber", "iron", "scabbard"
]


def _detect_synthetic_or_document(img_pil: Image.Image) -> tuple[bool, str]:
    """Detects artificial graphics, business cards, text slides, and non-biological flat images."""
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Check background pixel flatness
    vals, counts = np.unique(gray, return_counts=True)
    max_freq = float(np.max(counts)) / float(gray.size)

    # 2. Check flesh/organic meat chromatic distribution in HSV space
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    meat_mask = (((h <= 28) | (h >= 155)) & (s >= 25) & (v >= 30)) | ((h <= 35) & (s >= 18) & (v >= 50))
    flesh_ratio = float(np.sum(meat_mask)) / float(gray.size)

    if max_freq > 0.40 and flesh_ratio < 0.05:
        return True, "Non-Meat Image Detected: Digital Graphic / Card / Document identified. Please upload a valid raw meat sample."

    std_color = np.std(img_np.astype(float), axis=(0, 1))
    if float(np.max(std_color)) < 10.0 and flesh_ratio < 0.03:
        return True, "Non-Meat Image Detected: Blank or Monotone image identified. Please upload a valid meat sample."

    return False, ""


def validate_image(image_path: str, species_confidence: float) -> dict:
    """
    Validates whether an uploaded image is a valid meat/food sample
    or an invalid non-meat sample (e.g. business card, document, vehicle, electronics, pet).
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Step 1: Check for synthetic/document/card/blank uploads
        is_synth, synth_msg = _detect_synthetic_or_document(img)
        if is_synth:
            return {
                "is_valid": False,
                "error_code": "NON_MEAT_IMAGE",
                "message": synth_msg,
                "detected_category": "Digital Graphic / Document",
                "confidence": 99.0,
            }

        # Step 2: Semantic ImageNet object classification
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = imagenet_model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

        top_cat = categories[top5_catid[0].item()]
        top_conf = round(top5_prob[0].item() * 100, 2)
        top_cat_lower = top_cat.lower()

        # Check for allowed food packaging / butcher contexts
        for allowed in ALLOWED_FOOD_CONTEXTS:
            if allowed in top_cat_lower:
                return {
                    "is_valid": True,
                    "error_code": None,
                    "message": "Valid meat image",
                    "detected_category": top_cat,
                    "confidence": top_conf,
                }

        # Check against non-food categories
        for kw in EXPLICIT_NON_FOOD_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", top_cat_lower):
                if top_conf > 12.0:
                    return {
                        "is_valid": False,
                        "error_code": "NON_MEAT_IMAGE",
                        "message": f"Non-Meat Image Detected: Uploaded image identified as '{top_cat}' ({top_conf}% confidence). Please upload a valid meat sample.",
                        "detected_category": top_cat,
                        "confidence": top_conf,
                    }

        return {
            "is_valid": True,
            "error_code": None,
            "message": "Valid meat image",
            "detected_category": top_cat,
            "confidence": top_conf,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_code": "CORRUPT_IMAGE",
            "message": f"Unable to process image file: {str(e)}",
            "detected_category": "Unknown",
            "confidence": 0.0,
        }

