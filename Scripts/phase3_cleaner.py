import os
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm

EXTRACT_DIR = Path(r"d:\MeatVision_Project\Extracted_Datasets")
REPORTS_DIR = Path(r"d:\MeatVision_Project\Reports")

# Thresholds
MIN_RESOLUTION = (100, 100)  # Minimum width and height
BLUR_THRESHOLD = 50.0  # Minimum variance of Laplacian (lower means blurrier)


def is_blurry(image_path):
    try:
        # Load image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False  # Let the corrupt check handle it

        # Calculate Variance of Laplacian
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance < BLUR_THRESHOLD
    except Exception:
        return False


def clean_dataset():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "phase3_clean_report.txt"

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    print("Gathering image paths...")
    image_paths = set()
    for ext in extensions:
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext}"))
        image_paths.update(EXTRACT_DIR.rglob(f"*{ext.upper()}"))

    image_paths = list(image_paths)
    print(f"Found {len(image_paths)} images to inspect for quality.")

    deleted_corrupt = 0
    deleted_tiny = 0
    deleted_blurry = 0

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== Quality Filtering Report ===\n")
        f.write(f"Total Scanned: {len(image_paths)}\n\n")

        for img_path in tqdm(image_paths, desc="Filtering Bad Images"):
            try:
                # 1. Corrupt / Size Check
                with Image.open(img_path) as img:
                    img.verify()  # Verify it is a valid image

                with Image.open(img_path) as img:
                    w, h = img.size
                    if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
                        img_path.unlink()
                        f.write(f"DELETED (Tiny Resolution {w}x{h}): {img_path}\n")
                        deleted_tiny += 1
                        continue

                # 2. Blur Check
                if is_blurry(img_path):
                    img_path.unlink()
                    f.write(f"DELETED (Blurry): {img_path}\n")
                    deleted_blurry += 1

            except Exception as e:
                # Corrupt image
                try:
                    img_path.unlink()
                    f.write(f"DELETED (Corrupt): {img_path} - {e}\n")
                    deleted_corrupt += 1
                except:
                    pass

    print("\n=== Phase 3 Complete ===")
    print(f"Deleted Corrupt: {deleted_corrupt}")
    print(f"Deleted Tiny: {deleted_tiny}")
    print(f"Deleted Blurry: {deleted_blurry}")
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    print("=== Phase 3: Bad Image Filtering ===")
    clean_dataset()
