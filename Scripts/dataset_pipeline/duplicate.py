from pathlib import Path
import hashlib

from config import OUTPUT_DATASET, IMAGE_EXTENSIONS

print("=" * 80)
print("REMOVING EXACT DUPLICATES")
print("=" * 80)

hashes = {}
duplicates = 0
unique = 0

for image in OUTPUT_DATASET.rglob("*"):

    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:

        with open(image, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if file_hash in hashes:
            image.unlink()
            duplicates += 1
        else:
            hashes[file_hash] = image
            unique += 1

    except Exception:
        continue

print()
print("Unique Images :", unique)
print("Duplicates    :", duplicates)
print()
print("Finished.")
