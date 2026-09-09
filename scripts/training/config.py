import os
from pathlib import Path
import torch

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if os.path.exists("/kaggle/input"):
    base_input = Path("/kaggle/input")
    try:
        DATASET = next(base_input.rglob("Dataset"))
    except StopIteration:
        DATASET = base_input
else:
    DATASET = PROJECT_ROOT / "Dataset"

SPECIES_DIR = DATASET / "species"
FRESHNESS_DIR = DATASET / "freshness"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

GRAPH_DIR = RESULTS_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

REPORT_DIR = RESULTS_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True, parents=True)

# =====================================================
# TRAINING HYPERPARAMETERS
# =====================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device :", DEVICE)
