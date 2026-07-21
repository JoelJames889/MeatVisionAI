import os
from pathlib import Path
import shutil
from tqdm import tqdm

DATASET_ROOT = Path(r"C:\Users\cyber\Downloads\freshness_dataset")
OUTPUT_ROOT = Path(r"d:\MeatVision_Project\Clean_Dataset\Freshness")

CLASS_MAPPING = {"0": "fresh", "1": "halffresh", "2": "spoiled"}


def parse_and_move():
    splits = ["train", "valid", "test"]

    for split in splits:
        labels_dir = DATASET_ROOT / split / "labels"
        images_dir = DATASET_ROOT / split / "images"

        if not labels_dir.exists():
            continue

        label_files = list(labels_dir.glob("*.txt"))

        for label_file in tqdm(label_files, desc=f"Parsing {split}"):
            with open(label_file, "r") as f:
                content = f.read().strip()
                if not content:
                    continue
                # YOLOv8 format: class x_center y_center width height
                class_id = content.split(" ")[0]

            if class_id in CLASS_MAPPING:
                folder_name = CLASS_MAPPING[class_id]
                target_dir = OUTPUT_ROOT / folder_name
                target_dir.mkdir(parents=True, exist_ok=True)

                # Image could be jpg, jpeg, png
                image_name = label_file.stem
                found = False
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                    img_path = images_dir / f"{image_name}{ext}"
                    if img_path.exists():
                        dest_path = target_dir / f"{image_name}{ext}"
                        shutil.copy2(img_path, dest_path)
                        found = True
                        break

                if not found:
                    pass


if __name__ == "__main__":
    print("=== Parsing YOLOv8 Freshness Dataset ===")
    parse_and_move()
    print("Done!")
