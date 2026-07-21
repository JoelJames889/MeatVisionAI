from pathlib import Path
import shutil
import uuid

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

RAW = PROJECT_ROOT / "Extracted_Datasets"

OUTPUT = PROJECT_ROOT / "Unified_Dataset" / "freshness"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KEYWORDS = {
    "fresh": ["fresh"],
    "half_fresh": ["half", "half-fresh", "half_fresh"],
    "spoiled": ["spoiled", "rotten", "bad", "stale"],
}


def copy_images(folder, label):

    destination = OUTPUT / label
    destination.mkdir(parents=True, exist_ok=True)

    count = 0

    for file in folder.rglob("*"):

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        new_name = f"{label}_{uuid.uuid4().hex}{file.suffix.lower()}"

        shutil.copy2(file, destination / new_name)

        count += 1

    return count


print("=" * 80)
print("Building Freshness Dataset")
print("=" * 80)

total = 0

for dataset in RAW.iterdir():

    if not dataset.is_dir():
        continue

    for folder in dataset.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.lower()

        matched = False

        for label in KEYWORDS:

            if matched:
                break

            for keyword in KEYWORDS[label]:

                if keyword in name:

                    copied = copy_images(folder, label)

                    print(f"{folder.name:<35} ---> {label:<12} {copied}")

                    total += copied

                    matched = True

                    break

print()
print("=" * 80)
print("Finished")
print(f"Total Freshness Images : {total}")
