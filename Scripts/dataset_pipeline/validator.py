from PIL import Image

from config import OUTPUT_DATASET, IMAGE_EXTENSIONS

print("=" * 80)
print("VALIDATING DATASET")
print("=" * 80)

valid = 0
corrupt = 0

for image in OUTPUT_DATASET.rglob("*"):

    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:

        img = Image.open(image)
        img.verify()

        valid += 1

    except Exception:

        corrupt += 1

        try:
            image.unlink()
        except:
            pass

print()
print("Valid Images :", valid)
print("Corrupted    :", corrupt)
print()
print("Finished.")
