from pathlib import Path
import shutil

ROOT = Path(r"D:\MeatVision_Project")

SOURCE = ROOT / "Unified_Datasets"

DEST = ROOT / "Final_Dataset"

SPECIES = ["beef", "chicken", "fish", "pork"]

for sp in SPECIES:

    src = SOURCE / sp

    if not src.exists():
        continue

    dst = DEST / sp
    dst.mkdir(parents=True, exist_ok=True)

    count = 0

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:

        for img in src.rglob(ext):

            shutil.copy2(img, dst / img.name)

            count += 1

    print(f"{sp.upper()} : {count}")

print("\nFinished.")
