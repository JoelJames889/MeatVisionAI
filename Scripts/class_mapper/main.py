from pathlib import Path

ROOT = Path(r"D:\MeatVision_Project\Unified_Datasets")

for species in ROOT.iterdir():
    if not species.is_dir():
        continue

    print("\n==========================")
    print(species.name.upper())
    print("==========================")

    count = 0

    for img in species.rglob("*"):
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            count += 1

    print("Images :", count)
