from pathlib import Path
import hashlib

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

DATASET = PROJECT_ROOT / "Unified_Dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hashes = {}
duplicates = []

print("=" * 80)
print("Searching for Exact Duplicate Images")
print("=" * 80)

for image in DATASET.rglob("*"):

    if not image.is_file():
        continue

    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:
        with open(image, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if file_hash in hashes:
            duplicates.append(image)
        else:
            hashes[file_hash] = image

    except Exception:
        continue

print()
print("Unique Images :", len(hashes))
print("Duplicates    :", len(duplicates))

for img in duplicates:
    img.unlink()

print()
print("Duplicate removal completed.")
