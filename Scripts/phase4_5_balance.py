import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

EXTRACT_DIR = Path(r"d:\MeatVision_Project\Extracted_Datasets")
CLEAN_DIR = Path(r"d:\MeatVision_Project\Clean_Dataset")
REPORTS_DIR = Path(r"d:\MeatVision_Project\Reports")

# Targets
TARGET_COUNT = 3000

# Classes
SPECIES_CLASSES = ["beef", "chicken", "fish", "pork"]
FRESHNESS_CLASSES = ["fresh", "halffresh", "spoiled"]


def categorize_images():
    print("Categorizing available images...")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = set()
    for ext in extensions:
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext}"))
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext.upper()}"))

    image_paths = list(image_paths)

    species_dict = {cls: [] for cls in SPECIES_CLASSES}
    freshness_dict = {cls: [] for cls in FRESHNESS_CLASSES}

    for img_path in tqdm(image_paths, desc="Parsing Labels"):
        path_lower = str(img_path).lower()

        # Species
        for s_cls in SPECIES_CLASSES:
            if s_cls in path_lower:
                species_dict[s_cls].append(img_path)
                break

        # Freshness (Check halffresh first to avoid matching 'fresh' inside 'half-fresh')
        if "half" in path_lower or "semi" in path_lower:
            freshness_dict["halffresh"].append(img_path)
        elif "spoiled" in path_lower or "rotten" in path_lower:
            freshness_dict["spoiled"].append(img_path)
        elif "fresh" in path_lower:
            freshness_dict["fresh"].append(img_path)

    return species_dict, freshness_dict


def balance_and_copy(category_dict, category_name):
    print(f"\n=== Balancing {category_name} Dataset ===")
    out_dir = CLEAN_DIR / category_name

    with open(
        REPORTS_DIR / f"phase45_{category_name}_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(f"=== {category_name} Dataset Balancing Report ===\n\n")

        for cls, paths in category_dict.items():
            f.write(f"Class: {cls} - Available: {len(paths)}\n")
            print(f"Class: {cls.upper()} - Available: {len(paths)}")

            # Shuffle paths randomly
            random.shuffle(paths)

            # Sample target count
            if len(paths) >= TARGET_COUNT:
                sampled_paths = paths[:TARGET_COUNT]
                status = "SUCCESS (Full target reached)"
            else:
                sampled_paths = paths
                status = (
                    f"WARNING (Only {len(paths)} available, target is {TARGET_COUNT})"
                )
                print(f"  -> {status}")

            f.write(f"  -> Sampled: {len(sampled_paths)} - {status}\n")

            # Create destination folder
            cls_out_dir = out_dir / cls
            cls_out_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for i, img_path in enumerate(tqdm(sampled_paths, desc=f"Copying {cls}")):
                ext = img_path.suffix
                dest_path = cls_out_dir / f"{cls}_{i:04d}{ext}"
                try:
                    shutil.copy2(img_path, dest_path)
                except Exception as e:
                    f.write(f"  -> Failed to copy {img_path}: {e}\n")


if __name__ == "__main__":
    print("=== Phase 4 & 5: Dataset Rebuilding & Balancing ===")
    # Set seed for reproducibility
    random.seed(42)

    species_dict, freshness_dict = categorize_images()

    # Phase 4: Species
    balance_and_copy(species_dict, "Species")

    # Phase 5: Freshness
    balance_and_copy(freshness_dict, "Freshness")

    print("\nPhase 4 & 5 Complete! Check the Reports directory.")
