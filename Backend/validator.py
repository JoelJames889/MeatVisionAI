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

# Known ImageNet index ranges for pets
DOG_INDICES = set(range(151, 269))  # 118 dog breeds in ImageNet
CAT_INDICES = set(range(281, 288))  # Felines in ImageNet

# Allowed Food, Seafood, Culinary and Butchery contexts
ALLOWED_FOOD_CONTEXTS = [
    # Preparation & Kitchen Contexts
    "butcher shop", "plate", "tray", "cleaver", "frying pan", "wok", "platter", "bowl",
    "plastic bag", "cutting board", "chopping board", "dish", "tin", "foil", "table",
    # Food Preparations & Meats
    "meat loaf", "cheeseburger", "hotdog", "rotisserie", "carbonara", "dough", "hot pot",
    "ice cream", "trifle", "conch", "pomegranate", "head cabbage", "cauliflower", "squash",
    # Fish & Marine Food Contexts
    "sea lion", "seal", "gar", "barracouta", "coho", "puffer", "tench", "goldfish",
    "shark", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray",
    "eel", "salmon", "trout", "sturgeon", "anemone fish", "lionfish", "lobster", "crab",
    "crayfish", "king crab", "spiny lobster", "clam", "mussel", "oyster",
    "squid", "octopus", "nematode", "sea snake", "sea cucumber", "hog", "wild boar", "pig"
]

# Explicit non-food keywords covering open-world non-meat objects
NON_FOOD_PATTERNS = [
    # Documents, Digital Media & Office Items
    "web site", "envelope", "menu", "book jacket", "comic book", "binder", "notebook",
    "screen", "monitor", "television", "tv", "scoreboard", "billboard", "street sign", "poster",
    "paper towel", "toilet tissue", "rubber eraser", "ruler", "pencil box", "pencil sharpener",
    "ballpoint", "fountain pen", "quill", "packet", "carton", "digital clock", "cash machine", "slot",
    "safe", "matchstick", "lighter", "candle", "vase", "pottery", "pillow", "quilt", "curtain", "doormat",
    "book", "magazine", "newspaper", "passport", "ticket", "paper",
    # Vehicles & Transportation
    "car", "sports car", "convertible", "limousine", "minivan", "pickup", "cab", "racer",
    "automobile", "truck", "bus", "airplane", "aircraft", "bicycle", "motorcycle",
    "train", "boat", "ship", "submarines", "tractor", "trailer", "van", "jeep", "airliner",
    "warplane", "helicopter", "speedboat", "canoe", "kayak", "sailboat", "yacht", "moped",
    # Electronics & Appliances
    "laptop", "computer", "keyboard", "mouse", "telephone", "phone", "cellular", "camera",
    "radio", "speaker", "headphone", "modem", "refrigerator", "microwave", "clock", "watch",
    "smartwatch", "calculator", "remote control", "joystick", "loudspeaker", "vacuum", "iron",
    # Furniture & Household
    "sofa", "couch", "chair", "desk", "lamp", "bed", "wardrobe", "bookcase", "bathtub", "toilet",
    "closet", "cabinet", "dining table", "folding chair", "rocking chair", "studio couch", "toilet seat",
    # Apparel & Personal Items
    "shoe", "boot", "sneaker", "sandal", "glove", "hat", "cap", "jacket", "coat",
    "shirt", "trousers", "pants", "dress", "skirt", "sock", "tie", "umbrella",
    "sunglasses", "backpack", "handbag", "wallet", "suit", "trench coat", "cardigan", "sweatshirt",
    "jean", "bow tie", "scary", "wig", "jersey", "brassiere", "gown", "kimono",
    # Animals (Non-Food Pets / Wildlife)
    "dog", "puppy", "hound", "terrier", "retriever", "spaniel", "shepherd", "poodle", "collie",
    "cat", "kitten", "tabby", "persian", "siamese", "lynx", "cougar",
    "horse", "colt", "foal", "zebra", "elephant", "giraffe", "camel", "bear", "polar bear",
    "lion", "tiger", "cheetah", "leopard", "jaguar", "panther", "wolf", "fox", "coyote",
    "monkey", "chimpanzee", "gorilla", "baboon", "orangutan", "panda", "koala", "kangaroo",
    "deer", "elk", "moose", "antelope", "gazelle", "rabbit", "hare", "squirrel", "hedgehog",
    "penguin", "ostrich", "eagle", "hawk", "falcon", "owl", "parrot", "flamingo", "pelican",
    # Architecture & Scenery
    "building", "house", "church", "bridge", "tower", "castle", "skyscraper", "traffic light",
    "mountain", "alp", "volcano", "lakeside", "promontory", "cliff", "canyon", "valley",
    "palace", "monastery", "mosque", "cathedral", "barn", "lighthouse", "pier", "dam",
    # Sports & Instruments
    "ball", "football", "soccer", "basketball", "baseball", "tennis", "guitar", "piano", "violin",
    "drum", "flute", "trumpet", "saxophone", "racket", "skateboard", "ski", "surfboard", "kite",
    # Flora & Plants
    "daisy", "sunflower", "rose", "cardoon", "tree", "plant", "flower"
]

NON_FOOD_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in NON_FOOD_PATTERNS) + r")\b",
    re.IGNORECASE,
)


def _detect_synthetic_or_document(img_pil: Image.Image) -> tuple[bool, str]:
    """Detects artificial graphics, business cards, text slides, and non-biological flat images."""
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    total_pixels = float(gray.size)

    # 1. Check flesh/organic meat chromatic distribution in HSV space
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Red/pink/brown flesh tones + pale poultry/salmon tones
    meat_mask = (
        (((h <= 28) | (h >= 155)) & (s >= 18) & (v >= 25)) |
        ((h <= 45) & (s >= 15) & (v >= 40))
    )
    flesh_ratio = float(np.sum(meat_mask)) / total_pixels

    # 2. Check for blank / monotone images lacking flesh tones (e.g. solid white, black, blue, grey)
    std_color = np.std(img_np.astype(float), axis=(0, 1))
    if float(np.max(std_color)) < 10.0 and flesh_ratio < 0.03:
        return True, "Non-Meat Image Detected: Blank or Monotone image identified. Please upload a valid meat sample."

    # 3. Dominant non-meat colors (unnatural for raw meat)
    blue_mask = (h >= 90) & (h <= 135) & (s >= 30) & (v >= 35)
    blue_ratio = float(np.sum(blue_mask)) / total_pixels

    green_mask = (h >= 45) & (h <= 85) & (s >= 30) & (v >= 35)
    green_ratio = float(np.sum(green_mask)) / total_pixels

    if blue_ratio > 0.20 and flesh_ratio < 0.05:
        return True, f"Non-Meat Image Detected: Uploaded image contains dominant unnatural blue coloration ({round(blue_ratio*100, 1)}%). Real meat does not exhibit blue pigments."

    if green_ratio > 0.20 and flesh_ratio < 0.05:
        return True, f"Non-Meat Image Detected: Uploaded image contains dominant green foliage/nature ({round(green_ratio*100, 1)}%). Please upload a valid meat sample."

    # 4. Check for uniform digital graphic slides
    vals, counts = np.unique(gray, return_counts=True)
    max_freq = float(np.max(counts)) / total_pixels

    if max_freq > 0.55 and flesh_ratio < 0.02:
        return True, "Non-Meat Image Detected: Digital Graphic / Card / Document identified. Please upload a valid raw meat sample."

    return False, ""


def validate_image(image_path: str, species_confidence: float) -> dict:
    """
    Validates whether an uploaded image is a valid meat/food sample
    or an invalid non-meat sample (e.g. vehicle, electronics, pet, architecture, random object).
    """
    try:
        img = Image.open(image_path).convert("RGB")

        # Step 1: Check for synthetic/document/card/blank uploads and unnatural colorings
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

        top_idx = top5_catid[0].item()
        top_cat = categories[top_idx]
        top_conf = round(top5_prob[0].item() * 100, 2)
        top5_cats = [categories[cid.item()].lower() for cid in top5_catid]

        # Check if ANY of the top-5 categories indicate food, kitchen, butchery, or marine life
        for cat_name in top5_cats:
            for allowed in ALLOWED_FOOD_CONTEXTS:
                if allowed in cat_name:
                    return {
                        "is_valid": True,
                        "error_code": None,
                        "message": "Valid meat image",
                        "detected_category": top_cat,
                        "confidence": top_conf,
                    }

        # Step 3: Rejection Rules for Non-Meat Objects
        # Rule A: Dogs & Cats (explicit ImageNet index ranges)
        if top_idx in DOG_INDICES and top_conf >= 12.0:
            return {
                "is_valid": False,
                "error_code": "NON_MEAT_IMAGE",
                "message": f"Non-Meat Image Detected: Domestic canine identified as '{top_cat}' ({top_conf}% confidence). Please upload a valid meat sample.",
                "detected_category": f"Domestic Dog ({top_cat})",
                "confidence": top_conf,
            }

        if top_idx in CAT_INDICES and top_conf >= 12.0:
            return {
                "is_valid": False,
                "error_code": "NON_MEAT_IMAGE",
                "message": f"Non-Meat Image Detected: Feline identified as '{top_cat}' ({top_conf}% confidence). Please upload a valid meat sample.",
                "detected_category": f"Domestic Cat ({top_cat})",
                "confidence": top_conf,
            }

        # Rule B: Known Non-Food Patterns (vehicles, electronics, clothes, buildings, landscapes, instruments)
        top_cat_lower = top_cat.lower()
        if NON_FOOD_REGEX.search(top_cat_lower) and top_conf >= 8.0:
            return {
                "is_valid": False,
                "error_code": "NON_MEAT_IMAGE",
                "message": f"Non-Meat Image Detected: Uploaded image identified as '{top_cat}' ({top_conf}% confidence). Please upload a valid meat sample.",
                "detected_category": top_cat,
                "confidence": top_conf,
            }

        # Rule C: Organic Flesh Verification
        img_np = np.array(img)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        meat_mask = (
            (((h <= 28) | (h >= 155)) & (s >= 18) & (v >= 25)) |
            ((h <= 45) & (s >= 15) & (v >= 40))
        )
        flesh_ratio = float(np.sum(meat_mask)) / float(img_np.shape[0] * img_np.shape[1])

        if flesh_ratio < 0.03:
            return {
                "is_valid": False,
                "error_code": "NON_MEAT_IMAGE",
                "message": f"Non-Meat Image Detected: Object identified as '{top_cat}' ({top_conf}% confidence) with no organic meat tissue detected.",
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
