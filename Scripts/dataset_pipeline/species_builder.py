from pathlib import Path

from .config import UNIFIED_DATASET


def ensure_species_folders():
    species_root = UNIFIED_DATASET / "species"
    for name in ["beef", "chicken", "fish", "pork"]:
        (species_root / name).mkdir(parents=True, exist_ok=True)
