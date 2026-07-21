from pathlib import Path
import shutil

from config import SPECIES_DATASET, IMAGE_EXTENSIONS, SPECIES_KEYWORDS


class DatasetCopier:

    def __init__(self):

        SPECIES_DATASET.mkdir(exist_ok=True)

        for cls in SPECIES_KEYWORDS.keys():
            (SPECIES_DATASET / cls).mkdir(exist_ok=True)

    def detect_species(self, path: Path):

        text = str(path).lower()

        for species, keywords in SPECIES_KEYWORDS.items():

            for word in keywords:

                if word in text:
                    return species

        return None

    def copy_dataset(self, datasets):

        total = 0

        for dataset in datasets:

            print(f"\nScanning : {dataset['name']}")

            for image in dataset["path"].rglob("*"):

                if image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                species = self.detect_species(image)

                if species is None:
                    continue

                destination = SPECIES_DATASET / species

                filename = f"{species}_{total}{image.suffix.lower()}"

                shutil.copy2(image, destination / filename)

                total += 1

        print("\n===================================")
        print("Species Dataset Created Successfully")
        print("Images Copied :", total)
        print("===================================")
