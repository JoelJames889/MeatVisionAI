from pathlib import Path
from config import RAW_DATASET, IMAGE_EXTENSIONS


class DatasetScanner:

    def __init__(self):
        self.datasets = []

    def scan(self):

        print("=" * 80)
        print("Scanning Datasets...")
        print("=" * 80)

        for dataset in RAW_DATASET.iterdir():

            if not dataset.is_dir():
                continue

            image_count = 0

            for file in dataset.rglob("*"):

                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    image_count += 1

            self.datasets.append(
                {"name": dataset.name, "path": dataset, "images": image_count}
            )

        return self.datasets


if __name__ == "__main__":

    scanner = DatasetScanner()

    datasets = scanner.scan()

    print()

    total = 0

    for ds in datasets:

        print(f"{ds['name']:<55} {ds['images']:>8}")

        total += ds["images"]

    print("-" * 80)
    print(f"TOTAL IMAGES : {total}")
