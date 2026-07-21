from pathlib import Path
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROOT = Path(r"D:\MeatVision_Project\Extracted_Datasets")


def get_classes(folder):
    classes = set()

    for split in ["train", "valid", "test"]:

        split_path = folder / split

        if split_path.exists():

            for cls in split_path.iterdir():

                if cls.is_dir():
                    classes.add(cls.name)

    if len(classes) == 0:

        for item in folder.iterdir():

            if item.is_dir():

                has_image = False

                for f in item.rglob("*"):

                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        has_image = True
                        break

                if has_image:
                    classes.add(item.name)

    return sorted(classes)


def count_images(folder):

    count = 0

    for f in folder.rglob("*"):

        if f.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    return count


rows = []

for dataset in ROOT.iterdir():

    if not dataset.is_dir():
        continue

    rows.append(
        {
            "Dataset": dataset.name,
            "Images": count_images(dataset),
            "Classes": ", ".join(get_classes(dataset)),
        }
    )

df = pd.DataFrame(rows)

print(df)

df.to_csv("dataset_audit.csv", index=False)

print("\nAudit report saved as dataset_audit.csv")
