import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = MobileNet_V3_Small_Weights.DEFAULT
imagenet_model = models.mobilenet_v3_small(weights=weights).eval().to(DEVICE)
categories = weights.meta["categories"]
transform = weights.transforms()

import re

FOOD_MEAT_KEYWORDS = [
    "meat", "steak", "beef", "pork", "bacon", "ham", "sausage", "chicken",
    "poultry", "hen", "rooster", "turkey", "duck", "goose", "fish", "salmon",
    "trout", "tuna", "crab", "lobster", "shrimp", "seafood", "dish", "food",
    "plate", "carcass", "butcher", "pig", "hog", "boar", "cow", "bull", "ox",
    "lamb", "sheep", "potpie", "meatloaf", "hotdog", "cheeseburger", "hamburger",
    "grill", "barbecue", "frying_pan", "skillet", "roast", "noodle", "soup",
    "pizza", "sandwich", "burrito", "taco", "consomme", "trifle"
]


def validate_image(image_path: str, species_confidence: float) -> dict:
    """
    Validates whether an uploaded image is a valid meat/food sample
    or an invalid/non-meat image (e.g. vehicles, animals, gadgets, low confidence).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = imagenet_model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

        top_cat = categories[top5_catid[0].item()]
        top_conf = round(top5_prob[0].item() * 100, 2)

        top5_categories = [categories[idx.item()] for idx in top5_catid]

        food_match = None
        for cat in top5_categories:
            cat_lower = cat.lower()
            for kw in FOOD_MEAT_KEYWORDS:
                if re.search(r'\b' + re.escape(kw) + r'\b', cat_lower):
                    food_match = cat
                    break
            if food_match:
                break

        # 1. Non-food object detected with high confidence (e.g. sports car, laptop, dog, shoe)
        if not food_match and top_conf > 25.0:
            return {
                "is_valid": False,
                "error_code": "NON_MEAT_IMAGE",
                "message": f"Non-Meat Image Detected: Uploaded image appears to be '{top_cat}' ({top_conf}% confidence). Please upload a valid meat sample.",
                "detected_category": top_cat,
                "confidence": top_conf,
            }

        # 2. Low confidence across species predictions (< 55%) and no food match
        if not food_match and species_confidence < 55.0:
            return {
                "is_valid": False,
                "error_code": "UNRECOGNIZED_CONTENT",
                "message": "Invalid Image: Unrecognized image content or low confidence. Please upload a clear photo of raw or cooked meat.",
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
