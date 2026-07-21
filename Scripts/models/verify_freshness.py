import torch
from model import SpeciesModel
from config import *

model = SpeciesModel(3).get()

model.load_state_dict(
    torch.load(MODEL_DIR / "freshness_model.pth", map_location=DEVICE)
)

model.eval()

print("✅ Freshness model loaded successfully.")
