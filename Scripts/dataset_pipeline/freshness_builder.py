from pathlib import Path

from .config import UNIFIED_DATASET


def ensure_freshness_folders():
    freshness_root = UNIFIED_DATASET / "freshness"
    for name in ["fresh", "half_fresh", "spoiled"]:
        (freshness_root / name).mkdir(parents=True, exist_ok=True)
