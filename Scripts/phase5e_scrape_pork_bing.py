from bing_image_downloader import downloader
from pathlib import Path
import os
import shutil

TEMP_DIR = Path(r"d:\MeatVision_Project\temp_pork")
PORK_DIR = Path(r"d:\MeatVision_Project\Clean_Dataset\Species\pork")

queries = [
    "raw pork chop isolated white background",
    "raw pork belly meat",
    "raw pork tenderloin close up",
    "raw ground pork",
]


def run():
    print("=== Scraping Pork using Bing ===")
    PORK_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for q in queries:
        try:
            downloader.download(
                q,
                limit=40,
                output_dir=str(TEMP_DIR),
                adult_filter_off=False,
                force_replace=False,
                timeout=10,
                verbose=False,
            )
        except Exception as e:
            print(f"Failed to scrape {q}: {e}")

    # Move images to the pork directory
    count = 0
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                src = Path(root) / file

                # Make sure the filename is unique
                dest = PORK_DIR / f"bing_pork_{count}_{file}"

                try:
                    shutil.move(src, dest)
                    count += 1
                except:
                    pass

    # Cleanup temp directory
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print(f"\n=== Done! ===")
    print(f"Successfully scraped and downloaded {count} raw pork images into:")
    print(str(PORK_DIR))


if __name__ == "__main__":
    run()
