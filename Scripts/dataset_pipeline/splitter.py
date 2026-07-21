from pathlib import Path
import shutil
import random

from config import OUTPUT_DATASET, TRAIN_DIR, VALID_DIR, TEST_DIR, IMAGE_EXTENSIONS

random.seed(42)

# Remove old splits
for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

# Split both datasets
for dataset in ["species", "freshness"]:

    source_root = OUTPUT_DATASET / dataset

    if not source_root.exists():
        continue

    print("\n" + "=" * 70)
    print(dataset.upper())
    print("=" * 70)

    for cls in sorted(source_root.iterdir()):

        if not cls.is_dir():
            continue

        images = []

        for ext in IMAGE_EXTENSIONS:
            images.extend(cls.glob(f"*{ext}"))

        random.shuffle(images)

        total = len(images)

        train_end = int(total * 0.80)
        valid_end = int(total * 0.90)

        train = images[:train_end]
        valid = images[train_end:valid_end]
        test = images[valid_end:]

        for split_name, split_data in [
            ("train", train),
            ("valid", valid),
            ("test", test),
        ]:

            dst = OUTPUT_DATASET / split_name / dataset / cls.name
            dst.mkdir(parents=True, exist_ok=True)

            for img in split_data:
                shutil.copy2(img, dst / img.name)

        print(
            f"{cls.name:<15}"
            f"{total:>8}"
            f" | Train {len(train):>6}"
            f" | Valid {len(valid):>6}"
            f" | Test {len(test):>6}"
        )

print("\nFinished.")
