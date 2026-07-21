import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

EXTRACT_DIR = Path(r"d:\MeatVision_Project\Extracted_Datasets")
INSPECTION_DIR = Path(r"d:\MeatVision_Project\Inspection_Sample")

TARGET_COUNT = 100

SPECIES_CLASSES = ["beef", "chicken", "fish", "pork"]
FRESHNESS_CLASSES = ["fresh", "halffresh", "spoiled"]


def inspect():
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    print("Gathering image paths for inspection sample...")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = set()
    for ext in extensions:
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext}"))
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext.upper()}"))

    image_paths = list(image_paths)

    classes_dict = {cls: [] for cls in SPECIES_CLASSES + FRESHNESS_CLASSES}

    for img_path in tqdm(image_paths, desc="Parsing Labels"):
        path_lower = str(img_path).lower()

        # We put them in every applicable list
        for s_cls in SPECIES_CLASSES:
            if s_cls in path_lower:
                classes_dict[s_cls].append(img_path)

        if "half" in path_lower or "semi" in path_lower:
            classes_dict["halffresh"].append(img_path)
        elif "spoiled" in path_lower or "rotten" in path_lower:
            classes_dict["spoiled"].append(img_path)
        elif "fresh" in path_lower:
            classes_dict["fresh"].append(img_path)

    print("\nCopying 100 random images per class for visual inspection...")
    for cls, paths in classes_dict.items():
        if not paths:
            continue

        cls_dir = INSPECTION_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        random.shuffle(paths)
        sampled = paths[:TARGET_COUNT]

        for i, img in enumerate(sampled):
            try:
                dest = cls_dir / f"inspect_{i:03d}{img.suffix}"
                shutil.copy2(img, dest)
            except:
                pass

    print(
        f"\nDone! Please open {INSPECTION_DIR} in Windows Explorer to verify label quality."
    )


if __name__ == "__main__":
    inspect()
