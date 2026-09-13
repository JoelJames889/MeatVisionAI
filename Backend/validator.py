import re
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = MobileNet_V3_Small_Weights.DEFAULT
imagenet_model = models.mobilenet_v3_small(weights=weights).eval().to(DEVICE)
categories = weights.meta["categories"]
transform = weights.transforms()

# Explicit non-food object categories to reject (vehicles, electronics, apparel, pets, etc.)
EXPLICIT_NON_FOOD_KEYWORDS = [
    "car", "sports car", "convertible", "limousine", "minivan", "pickup", "cab", "racer",
    "automobile", "truck", "bus", "airplane", "aircraft", "bicycle", "motorcycle",
    "train", "boat", "ship", "submarines", "tractor", "trailer", "laptop", "computer", "monitor",
    "keyboard", "mouse", "telephone", "phone", "cellular", "television", "tv", "camera",
    "radio", "speaker", "headphone", "modem", "sofa", "couch", "chair", "desk", "lamp",
    "refrigerator", "microwave", "clock", "bed", "shoe", "boot", "sneaker", "sandal", "glove", "hat",
    "cap", "jacket", "coat", "shirt", "trousers", "pants", "dress", "skirt", "sock", "tie", "umbrella",
    "sunglasses", "backpack", "handbag", "wallet", "cat", "dog", "puppy", "kitten",
    "horse", "lion", "tiger", "bear", "elephant", "giraffe", "monkey", "penguin", "zebra",
    "building", "house", "church", "bridge", "tower", "castle", "skyscraper", "traffic light"
]


def validate_image(image_path: str, species_confidence: float) -> dict:
    """
    Validates whether an uploaded image is a valid meat/food sample
    or an explicit non-food object (e.g. vehicles, electronics, apparel, pets).
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

        # Check if top ImageNet category is explicitly a non-food object
        top_cat_lower = top_cat.lower()
        is_explicit_non_food = False
        for kw in EXPLICIT_NON_FOOD_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', top_cat_lower):
                is_explicit_non_food = True
                break

        # Only reject if ImageNet is confident in an explicitly non-food object (e.g. car, phone, cat)
        if is_explicit_non_food and top_conf > 30.0:
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
