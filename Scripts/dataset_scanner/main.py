from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\MeatVision_Project\Extracted_Datasets")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []

for dataset in ROOT.iterdir():

    if not dataset.is_dir():
        continue

    train = 0
    valid = 0
    test = 0
    total = 0
    yolo = False
    classification = False

    for f in dataset.rglob("*"):

        if f.suffix.lower() in IMAGE_EXT:

            total += 1

            p = str(f.parent).lower()

            if "train" in p:
                train += 1

            elif "valid" in p or "val" in p:
                valid += 1

            elif "test" in p:
                test += 1

        if f.suffix == ".txt":
            yolo = True

    if train == 0 and valid == 0 and test == 0:
        classification = True

    records.append(
        {
            "Dataset": dataset.name,
            "Images": total,
            "Train": train,
            "Valid": valid,
            "Test": test,
            "YOLO": yolo,
            "Classification": classification,
        }
    )

df = pd.DataFrame(records)

print(df)

out = Path(r"D:\MeatVision_Project\Documentation\Dataset_Report.xlsx")

df.to_excel(out, index=False)

print("\nDataset_Report.xlsx created successfully.")
