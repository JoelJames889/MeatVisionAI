from pathlib import Path
import shutil

PROJECT = Path(r"D:\MeatVision_Project")

RAW = PROJECT / "Extracted_Datasets"
DEST = PROJECT / "Unified_Dataset" / "species" / "beef"

DEST.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BEEF_KEYWORDS = ["beef", "cow"]

copied = 0
skipped = 0

print("=" * 80)
print("IMPORTING ALL BEEF IMAGES")
print("=" * 80)

for file in RAW.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    path = str(file).lower()

    if not any(word in path for word in BEEF_KEYWORDS):
        continue

    new_name = f"beef_{copied}_{file.name}"

    destination = DEST / new_name

    if destination.exists():
        skipped += 1
        continue

    shutil.copy2(file, destination)

    copied += 1

print()
print("=" * 80)
print("FINISHED")
print("=" * 80)

print("Copied :", copied)
print("Skipped:", skipped)
