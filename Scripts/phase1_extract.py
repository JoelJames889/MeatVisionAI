import os
import zipfile
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(r"d:\MeatVision_Project\Raw_Datasets")
EXTRACT_DIR = Path(r"d:\MeatVision_Project\Extracted_Datasets")


def extract_all():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = list(RAW_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files in {RAW_DIR}")

    for zip_file in zip_files:
        extract_path = EXTRACT_DIR / zip_file.stem

        # Skip if already extracted
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"Skipping {zip_file.name} - already extracted.")
            continue

        print(f"Extracting {zip_file.name} to {extract_path}...")
        extract_path.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Use tqdm for a progress bar if the zip has many files
                for member in tqdm(
                    zip_ref.namelist(), desc=f"Extracting {zip_file.stem}"
                ):
                    zip_ref.extract(member, extract_path)
            print(f"Successfully extracted {zip_file.name}")
        except zipfile.BadZipFile:
            print(f"ERROR: {zip_file.name} is a bad zip file and cannot be extracted.")
        except Exception as e:
            print(f"ERROR: Failed to extract {zip_file.name}. Reason: {e}")


if __name__ == "__main__":
    print("=== Phase 1: Dataset Extraction ===")
    extract_all()
    print("Extraction phase complete!")
