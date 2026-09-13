import os
from pathlib import Path
import torch

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _find_dataset_root():
    # 1. Check explicit environment variable
    if "DATASET_DIR" in os.environ and Path(os.environ["DATASET_DIR"]).exists():
        return Path(os.environ["DATASET_DIR"])

    # 2. Check local project Dataset directory
    local_dataset = PROJECT_ROOT / "Dataset"
    if (local_dataset / "species").exists():
        return local_dataset

    # 3. Check Google Drive Dataset paths
    drive_paths = [
        Path("/content/drive/MyDrive/Dataset"),
        Path("/content/drive/MyDrive/MeatVisionAI/Dataset"),
        Path("/content/drive/MyDrive/MeatVision_Project/Dataset"),
    ]
    for dp in drive_paths:
        if (dp / "species").exists():
            return dp

    # 4. Check Colab /content/Dataset
    colab_paths = [
        Path("/content/Dataset"),
        Path("/content/dataset"),
        Path("/content/MeatVisionAI/Dataset"),
        Path("/content/MeatVision_Project/Dataset"),
    ]
    for cp in colab_paths:
        if (cp / "species").exists():
            return cp

    # 5. Check Kaggle input directory ONLY IF it actually contains species/freshness data
    if os.path.exists("/kaggle/input"):
        base_input = Path("/kaggle/input")
        matches = list(base_input.rglob("species"))
        if matches:
            return matches[0].parent
        dataset_matches = list(base_input.rglob("Dataset"))
        if dataset_matches and (dataset_matches[0] / "species").exists():
            return dataset_matches[0]

    # Fallback to local project Dataset directory
    return local_dataset


DATASET = _find_dataset_root()

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


def validate_dataset_exists(dir_path, dataset_name="species"):
    if not dir_path.exists() or not any(dir_path.iterdir()):
        raise FileNotFoundError(
            f"\n\n{'='*80}\n"
            f"❌ DATASET NOT FOUND AT: '{dir_path}'\n"
            f"{'='*80}\n"
            f"The training script requires the '{dataset_name}' images dataset.\n"
            f"Expected subfolders inside '{dir_path}' (e.g. beef, chicken, fish, pork).\n\n"
            f"How to fix in Google Colab:\n"
            f"  1. Mount Google Drive containing your Dataset folder:\n"
            f"     from google.colab import drive; drive.mount('/content/drive')\n"
            f"     Ensure your Drive has a folder named 'Dataset' with 'species' inside.\n"
            f"  2. OR Upload 'Dataset.zip' to /content/ and extract it:\n"
            f"     !unzip Dataset.zip -d /content/MeatVisionAI/\n"
            f"  3. OR Set the DATASET_DIR environment variable:\n"
            f"     import os; os.environ['DATASET_DIR'] = '/path/to/your/Dataset'\n"
            f"{'='*80}\n"
        )

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

