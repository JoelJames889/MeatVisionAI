from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetAnalyzer:

    def analyze(self, dataset):

        images = 0

        folders = []

        yaml_files = []

        txt_files = []

        for item in dataset.rglob("*"):

            if item.is_dir():
                folders.append(item.name)

            else:

                if item.suffix.lower() in IMAGE_EXTENSIONS:
                    images += 1

                elif item.suffix.lower() in [".yaml", ".yml"]:
                    yaml_files.append(item)

                elif item.suffix.lower() == ".txt":
                    txt_files.append(item)

        return {
            "dataset": dataset.name,
            "path": str(dataset),
            "images": images,
            "folders": folders,
            "yaml": yaml_files,
            "txt": txt_files,
        }
