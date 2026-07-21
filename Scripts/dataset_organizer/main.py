from pathlib import Path
import shutil

PROJECT = Path(r"D:\MeatVision_Project")

SOURCE = PROJECT / "Extracted_Datasets"
DEST = PROJECT / "Unified_Datasets"

SPECIES = {
    "chicken": ["Chicken", "chicken", "drumstick", "wing", "thigh", "breast"],
    "beef": ["Beef", "beef"],
    "fish": ["Fish", "fish", "tilapia", "salmon", "tuna"],
    "pork": ["Pork", "pork"],
    "mutton": ["mutton", "goat", "lamb", "sheep"],
}

for species in SPECIES:
    (DEST / species).mkdir(parents=True, exist_ok=True)

count = 0

for img in SOURCE.rglob("*"):

    if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        continue

    filename = img.name.lower()

    copied = False

    for species, keywords in SPECIES.items():

        if any(word.lower() in filename for word in keywords):

            shutil.copy2(img, DEST / species / img.name)

            copied = True
            count += 1
            break

print("=" * 50)
print("Finished")
print("Images Copied :", count)
print("=" * 50)
