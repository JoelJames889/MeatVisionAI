from pathlib import Path

# ==========================
# PROJECT PATHS
# ==========================

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

RAW_DATASETS = PROJECT_ROOT / "Extracted_Datasets"

OUTPUT_DATASET = PROJECT_ROOT / "Dataset"

SPECIES_OUTPUT = OUTPUT_DATASET / "species"

FRESHNESS_OUTPUT = OUTPUT_DATASET / "freshness"

TRAIN_DIR = OUTPUT_DATASET / "train"

VALID_DIR = OUTPUT_DATASET / "valid"

TEST_DIR = OUTPUT_DATASET / "test"

# ==========================
# IMAGE TYPES
# ==========================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==========================
# SPECIES KEYWORDS
# ==========================

SPECIES_KEYWORDS = {
    "beef": ["beef", "cow", "cattle"],
    "chicken": ["chicken", "breast", "wing", "leg", "thigh", "drumstick"],
    "fish": [
        "fish",
        "salmon",
        "tuna",
        "trout",
        "sea bass",
        "shrimp",
        "mackerel",
        "bream",
        "sprat",
    ],
    "pork": ["pork", "pig", "ham"],
}

# ==========================
# FRESHNESS KEYWORDS
# ==========================

FRESHNESS_KEYWORDS = {
    "fresh": ["fresh", "fresh meat", "meat_fresh", "poultry_fresh", "a1"],
    "half_fresh": ["half-fresh", "half_fresh", "half fresh", "a2"],
    "spoiled": [
        "spoiled",
        "rotten",
        "stale",
        "meat_spoiled",
        "poultry_spoiled",
        "a3",
        "a4",
        "a5",
    ],
}
