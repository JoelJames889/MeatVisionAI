from pathlib import Path

ROOT = Path(r"D:\MeatVision_Project\Extracted_Datasets")

print("=" * 80)

for file in ROOT.rglob("*"):

    if file.name.lower() in ["_classes.txt", "classes.txt", "classes.names"]:

        print("\nDataset :", file.parent)

        try:
            with open(file, "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            print(e)

print("=" * 80)
print("Finished")
