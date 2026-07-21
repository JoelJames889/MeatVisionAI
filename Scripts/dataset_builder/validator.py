from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"D:\MeatVision_Project")
DATASET = PROJECT_ROOT / "Unified_Dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

bad_images = 0
good_images = 0

print("=" * 80)
print("Validating Images")
print("=" * 80)

for image in DATASET.rglob("*"):

    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:
        with Image.open(image) as img:
            img.verify()

        good_images += 1

    except Exception:
        bad_images += 1
        print("Removing:", image.name)
        image.unlink()

print()
print("=" * 80)
print("Validation Completed")
print("=" * 80)
print("Valid Images :", good_images)
print("Corrupted    :", bad_images)
