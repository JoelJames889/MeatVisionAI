import random
import shutil
from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT = Path(r"D:\MeatVision_Project")

SOURCE = PROJECT / "Dataset"
DESTINATION = PROJECT / "Dataset_Balanced"

random.seed(42)

# ==========================================================
# Number of images to keep
# ==========================================================

TARGET = {"fresh": 20000, "half_fresh": None, "spoiled": 20000}  # Keep all

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

# ==========================================================
# COPY SPECIES DATASET AS-IS
# ==========================================================

print("=" * 70)
print("COPYING SPECIES DATASET")
print("=" * 70)

for split in ["train", "valid", "test"]:

    src = SOURCE / split / "species"
    dst = DESTINATION / split / "species"

    shutil.copytree(src, dst, dirs_exist_ok=True)

# ==========================================================
# BALANCE FRESHNESS TRAINING DATASET
# ==========================================================

print("\n" + "=" * 70)
print("BALANCING FRESHNESS TRAIN DATASET")
print("=" * 70)

train_source = SOURCE / "train" / "freshness"
train_destination = DESTINATION / "train" / "freshness"

for class_folder in train_source.iterdir():

    if not class_folder.is_dir():
        continue

    classname = class_folder.name

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(class_folder.glob(ext))

    images = sorted(images)

    print(f"\n{classname}")
    print(f"Original : {len(images)}")

    limit = TARGET[classname]

    if limit is not None and len(images) > limit:
        images = random.sample(images, limit)

    print(f"Keeping  : {len(images)}")

    output = train_destination / classname
    output.mkdir(parents=True, exist_ok=True)

    for img in images:
        shutil.copy2(img, output / img.name)

# ==========================================================
# COPY VALID FRESHNESS
# ==========================================================

print("\nCopying validation dataset...")

shutil.copytree(
    SOURCE / "valid" / "freshness",
    DESTINATION / "valid" / "freshness",
    dirs_exist_ok=True,
)

# ==========================================================
# COPY TEST FRESHNESS
# ==========================================================

print("Copying testing dataset...")

shutil.copytree(
    SOURCE / "test" / "freshness",
    DESTINATION / "test" / "freshness",
    dirs_exist_ok=True,
)

print("\n" + "=" * 70)
print("BALANCED DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print("\nLocation:")
print(DESTINATION)
