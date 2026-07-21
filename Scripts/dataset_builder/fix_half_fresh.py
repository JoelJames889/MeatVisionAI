from pathlib import Path
import shutil

PROJECT = Path(r"D:\MeatVision_Project")

RAW = PROJECT / "Extracted_Datasets"
DEST = PROJECT / "Unified_Dataset" / "freshness" / "half_fresh"

DEST.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KEYWORDS = [
    "half-fresh",
    "half_fresh",
    "half fresh",
]

copied = 0

print("=" * 70)
print("IMPORTING HALF-FRESH IMAGES")
print("=" * 70)

for file in RAW.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    path = str(file).lower()

    if any(keyword in path for keyword in KEYWORDS):

        name = f"half_{copied}_{file.name}"

        shutil.copy2(file, DEST / name)

        copied += 1

print(f"\nHalf-Fresh Images Imported : {copied}")
