from pathlib import Path
import zipfile

RAW = Path(r"D:\MeatVision_Project\Raw_Datasets")
OUT = Path(r"D:\MeatVision_Project\Extracted_Datasets")

OUT.mkdir(exist_ok=True)

for zip_file in RAW.glob("*.zip"):
    folder = OUT / zip_file.stem

    if folder.exists():
        print(f"Skipped : {zip_file.name}")
        continue

    print(f"Extracting : {zip_file.name}")

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(folder)

print("\nAll datasets extracted successfully!")
