from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

DATASET = PROJECT_ROOT / "Unified_Dataset"

OUTPUT = PROJECT_ROOT / "Results"

OUTPUT.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []

print("=" * 80)
print("Generating Metadata...")
print("=" * 80)

for species_folder in DATASET.iterdir():

    if not species_folder.is_dir():
        continue

    species = species_folder.name.lower()

    for image in species_folder.rglob("*"):

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        name = image.stem.lower()

        freshness = "unknown"

        if "fresh" in name:
            freshness = "fresh"

        if "half" in name:
            freshness = "half_fresh"

        if "spoiled" in name:
            freshness = "spoiled"

        if "rotten" in name:
            freshness = "spoiled"

        if "bad" in name:
            freshness = "spoiled"

        records.append(
            {"image": str(image), "species": species, "freshness": freshness}
        )

df = pd.DataFrame(records)

df.to_csv(OUTPUT / "metadata.csv", index=False)

print()
print("Metadata Created Successfully")
print(f"Images : {len(df)}")
print(df.head())
