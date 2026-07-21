from pathlib import Path
import shutil
import uuid

from scanner import DatasetScanner
from mapper import DatasetMapper

from config import SPECIES_OUTPUT, FRESHNESS_OUTPUT, IMAGE_EXTENSIONS


class DatasetBuilder:

    def __init__(self):

        self.scanner = DatasetScanner()
        self.mapper = DatasetMapper()

        self.species_count = {"beef": 0, "chicken": 0, "fish": 0, "pork": 0}

        self.freshness_count = {"fresh": 0, "half_fresh": 0, "spoiled": 0}

    def copy_image(self, image, destination):

        destination.mkdir(parents=True, exist_ok=True)

        new_name = uuid.uuid4().hex + image.suffix.lower()

        shutil.copy2(image, destination / new_name)

    def process_dataset(self, dataset):

        root = dataset["path"]

        print(f"\nProcessing : {root.name}")

        for image in root.rglob("*"):

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            species, freshness = self.mapper.classify(image)

            if species is not None:

                self.copy_image(image, SPECIES_OUTPUT / species)

                self.species_count[species] += 1

            if freshness is not None:

                self.copy_image(image, FRESHNESS_OUTPUT / freshness)

                self.freshness_count[freshness] += 1

    def build(self):

        datasets = self.scanner.scan()

        print("\n")
        print("=" * 80)
        print("BUILDING DATASET")
        print("=" * 80)

        for dataset in datasets:

            self.process_dataset(dataset)

        print("\n")
        print("=" * 80)
        print("SPECIES")
        print("=" * 80)

        for k, v in self.species_count.items():

            print(f"{k:<15}{v}")

        print()

        print("=" * 80)
        print("FRESHNESS")
        print("=" * 80)

        for k, v in self.freshness_count.items():

            print(f"{k:<15}{v}")

        print()

        print("Dataset Build Completed.")


if __name__ == "__main__":

    DatasetBuilder().build()
