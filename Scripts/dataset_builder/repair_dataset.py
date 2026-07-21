from pathlib import Path
import shutil

PROJECT = Path(r"D:\MeatVision_Project")

SPECIES = PROJECT / "Unified_Dataset" / "species"

print("=" * 70)
print("Repairing Dataset")
print("=" * 70)

# ----------------------------
# Fix Beef Folder
# ----------------------------

wrong_beef = SPECIES / "pork" / "beef"
correct_beef = SPECIES / "beef"

correct_beef.mkdir(parents=True, exist_ok=True)

moved = 0

if wrong_beef.exists():

    for img in wrong_beef.iterdir():

        if img.is_file():

            shutil.move(str(img), correct_beef / img.name)
            moved += 1

print(f"Moved Beef Images : {moved}")

# ----------------------------
# Remove Empty Folder
# ----------------------------

try:
    wrong_beef.rmdir()
except:
    pass

print("\nRepair Complete.")
