from pathlib import Path
import shutil

PROJECT = Path(r"D:\MeatVision_Project")

SOURCE = PROJECT / "Extracted_Datasets" / "Detect Pork.v7i.yolov8"
DEST = PROJECT / "Unified_Dataset" / "species" / "pork"

DEST.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

copied = 0

for split in ["train", "valid", "test"]:

    image_folder = SOURCE / split / "images"

    if not image_folder.exists():
        continue

    for image in image_folder.iterdir():

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        new_name = f"pork_{copied}_{image.name}"

        shutil.copy2(image, DEST / new_name)

        copied += 1

print("=" * 60)
print("Pork Dataset Imported")
print("=" * 60)
print("Images Copied :", copied)
