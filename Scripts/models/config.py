import os
from pathlib import Path
import torch

# =====================================================
# PROJECT PATHS (AUTO-DETECT KAGGLE)
# =====================================================

if os.path.exists("/kaggle/input"):
    PROJECT_ROOT = Path("/kaggle/working")
    base_input = Path("/kaggle/input")
    try:
        DATASET = next(base_input.rglob("Clean_Dataset"))
    except StopIteration:
        DATASET = base_input  # Fallback
else:
    PROJECT_ROOT = Path(r"D:\MeatVision_Project")
    DATASET = PROJECT_ROOT / "Clean_Dataset"

SPECIES_DIR = DATASET / "Species"
FRESHNESS_DIR = DATASET / "Freshness"

MODEL_DIR = PROJECT_ROOT / "Models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

SPECIES_MODEL_DIR = MODEL_DIR / "species"
FRESHNESS_MODEL_DIR = MODEL_DIR / "freshness"

RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

GRAPH_DIR = RESULTS_DIR / "Graphs"
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

REPORT_DIR = RESULTS_DIR / "Reports"
REPORT_DIR.mkdir(exist_ok=True, parents=True)

# =====================================================
# TRAINING
# =====================================================

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

# Lower workers for Kaggle compatibility
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device :", DEVICE)
