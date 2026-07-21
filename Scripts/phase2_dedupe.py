import os
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm
import collections

EXTRACT_DIR = Path(r"d:\MeatVision_Project\Extracted_Datasets")
REPORTS_DIR = Path(r"d:\MeatVision_Project\Reports")


def remove_duplicates():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "phase2_dedupe_report.txt"

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    print("Gathering image paths...")
    image_paths = set()
    for ext in extensions:
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext}"))
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext.upper()}"))

    image_paths = list(image_paths)
    total_images = len(image_paths)
    print(f"Found {total_images} images to process for deduplication.")

    hashes = {}
    duplicates = []

    # Process images and compute perceptual hash
    print("Computing perceptual hashes...")
    for img_path in tqdm(image_paths, desc="Hashing Images"):
        try:
            with Image.open(img_path) as img:
                # Calculate perceptual hash (phash)
                img_hash = str(imagehash.phash(img))

                if img_hash in hashes:
                    duplicates.append(img_path)
                else:
                    hashes[img_hash] = img_path
        except Exception as e:
            # If we can't open it, we'll let Phase 3 deal with corrupt images,
            # but we can safely ignore it here.
            pass

    print(f"\nFound {len(duplicates)} duplicate images.")

    # Delete duplicates
    print("Deleting duplicates...")
    deleted_count = 0
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== Duplicate Removal Report ===\n")
        f.write(f"Total Images Scanned: {total_images}\n")
        f.write(f"Duplicates Found: {len(duplicates)}\n\n")

        for dup in tqdm(duplicates, desc="Deleting"):
            try:
                dup.unlink()
                f.write(f"DELETED: {dup}\n")
                deleted_count += 1
            except Exception as e:
                f.write(f"FAILED TO DELETE: {dup} - {e}\n")

    print(f"Phase 2 Complete. Deleted {deleted_count} duplicate images.")
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    print("=== Phase 2: Duplicate Removal ===")
    remove_duplicates()
