from pathlib import Path

from config import RAW_DATASETS, IMAGE_EXTENSIONS


class DatasetScanner:

    def __init__(self):

        self.datasets = []

    def count_images(self, folder):

        total = 0

        for file in folder.rglob("*"):

            if file.suffix.lower() in IMAGE_EXTENSIONS:
                total += 1

        return total

    def scan(self):

        print("=" * 80)
        print("SCANNING DATASETS")
        print("=" * 80)

        self.datasets.clear()

        if not RAW_DATASETS.exists():

            raise FileNotFoundError(f"{RAW_DATASETS} does not exist.")

        for dataset in sorted(RAW_DATASETS.iterdir()):

            if not dataset.is_dir():
                continue

            info = {
                "name": dataset.name,
                "path": dataset,
                "images": self.count_images(dataset),
                "train": (dataset / "train").exists(),
                "valid": (dataset / "valid").exists(),
                "test": (dataset / "test").exists(),
            }

            self.datasets.append(info)

        print()

        print(f"Datasets Found : {len(self.datasets)}")

        total = sum(x["images"] for x in self.datasets)

        print(f"Total Images   : {total}")

        print()

        for ds in self.datasets:

            print(f"{ds['name']:<55}" f"{ds['images']:>8}")

        print()

        return self.datasets


if __name__ == "__main__":

    scanner = DatasetScanner()

    scanner.scan()
