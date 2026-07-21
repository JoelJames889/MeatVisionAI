from pathlib import Path

from .scanner import DatasetScanner
from .builder import build_dataset
from .mapper import map_dataset
from .duplicate import remove_duplicates
from .splitter import split_dataset


def main():
    print("Starting dataset pipeline...")
    scanner = DatasetScanner()
    scanner.scan()
    map_dataset()
    build_dataset()
    remove_duplicates(Path("."))
    split_dataset(Path("."))
    print("Dataset pipeline finished.")


if __name__ == "__main__":
    main()
