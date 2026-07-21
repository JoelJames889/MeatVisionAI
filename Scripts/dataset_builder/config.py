from pathlib import Path

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

RAW_DATASET = PROJECT_ROOT / "Extracted_Datasets"

SPECIES_DATASET = PROJECT_ROOT / "Unified_Dataset"

FRESHNESS_DATASET = PROJECT_ROOT / "Freshness_Dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPECIES_KEYWORDS = {
    "beef": ["beef", "cow"],
    "chicken": [
        "chicken",
        "breast",
        "wing",
        "leg",
        "thigh",
        "drumstick",
        "quarter",
        "quater",
    ],
    "fish": [
        "fish",
        "salmon",
        "tuna",
        "trout",
        "shrimp",
        "sea bass",
        "red mullet",
        "horse mackerel",
        "gilt head bream",
        "black sea sprat",
    ],
    "pork": ["pork", "fork", "pig", "swine", "ham", "bacon"],
}
