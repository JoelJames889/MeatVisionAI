import torch

from model import SpeciesModel
from config import *

print("=" * 60)
print("VERIFYING MODELS")
print("=" * 60)

# ==========================================================
# Species Model
# ==========================================================

species_path = MODEL_DIR / "species_model.pth"

print("Species Model Path :", species_path)

if species_path.exists():

    try:

        model = SpeciesModel(4).get()

        model.load_state_dict(torch.load(species_path, map_location=DEVICE))

        model.eval()

        print("✅ Species Model Loaded Successfully")

    except Exception as e:

        print("❌ Species Model Failed")
        print(e)

else:

    print("❌ Species Model Not Found")

# ==========================================================
# Freshness Model
# ==========================================================

freshness_path = MODEL_DIR / "freshness_model.pth"

print("\nFreshness Model Path :", freshness_path)

if freshness_path.exists():

    try:

        model = SpeciesModel(3).get()

        model.load_state_dict(torch.load(freshness_path, map_location=DEVICE))

        model.eval()

        print("✅ Freshness Model Loaded Successfully")

    except Exception as e:

        print("❌ Freshness Model Failed")
        print(e)

else:

    print("❌ Freshness Model Not Found")

print("=" * 60)
