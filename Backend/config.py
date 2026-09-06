from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

SPECIES_MODEL = MODEL_DIR / "species_model.pth"
FRESHNESS_MODEL = MODEL_DIR / "freshness_model.pth"

IMAGE_SIZE = 224
