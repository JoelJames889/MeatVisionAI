from pathlib import Path

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

MODEL_DIR = PROJECT_ROOT / "Models"

SPECIES_MODEL = MODEL_DIR / "species_model.pth"
FRESHNESS_MODEL = MODEL_DIR / "freshness_model.pth"

IMAGE_SIZE = 224
