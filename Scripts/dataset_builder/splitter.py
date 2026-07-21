from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

PROJECT = Path(r"D:\MeatVision_Project")

SPECIES = PROJECT / "Unified_Dataset" / "species"
FRESHNESS = PROJECT / "Unified_Dataset" / "freshness"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(dataset_root):

    for cls in dataset_root.iterdir():

        if not cls.is_dir():
            continue

        images = []

        for img in cls.iterdir():

            if img.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(img)

        print(f"\n{cls.name}")
        print("Images :", len(images))

        if len(images) == 0:
            print("Skipping class because there are no image files.")
            continue

        if len(images) == 1:
            train = images
            valid = []
            test = []
        elif len(images) == 2:
            train, test = train_test_split(
                images, test_size=0.5, random_state=42, shuffle=True
            )
            valid = []
        else:
            train, temp = train_test_split(
                images, test_size=0.20, random_state=42, shuffle=True
            )

            if len(temp) > 1:
                valid, test = train_test_split(temp, test_size=0.50, random_state=42)
            else:
                valid = []
                test = temp

        for split_name, split_data in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:

            destination = dataset_root / split_name / cls.name

            destination.mkdir(parents=True, exist_ok=True)

            for img in split_data:

                shutil.copy2(img, destination / img.name)

        print(
            f"Train : {len(train)} | " f"Valid : {len(valid)} | " f"Test : {len(test)}"
        )


print("=" * 80)
print("Splitting Species Dataset")
print("=" * 80)

split_dataset(SPECIES)

print("\n")

print("=" * 80)
print("Splitting Freshness Dataset")
print("=" * 80)

split_dataset(FRESHNESS)

print("\nFinished")
