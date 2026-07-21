import os
import requests
import uuid
import time
from pathlib import Path
from duckduckgo_search import DDGS
from tqdm import tqdm

PORK_DIR = Path(r"d:\MeatVision_Project\Clean_Dataset\Species\pork")

SEARCH_TERMS = [
    "raw pork chop",
    "raw pork ribs",
    "raw pork belly",
    "raw pork tenderloin",
    "raw pork meat isolated",
    "raw ground pork",
    "raw pork shoulder",
    "raw pork loin",
    "raw diced pork",
    "fresh raw pork cuts",
]

MAX_RESULTS_PER_TERM = 50


def scrape_pork_meat():
    print(f"=== Phase 5d: Scraping Pork Images ===")
    PORK_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    success_count = 0
    with DDGS() as ddgs:
        for term in SEARCH_TERMS:
            print(f"\nSearching for: '{term}'...")
            try:
                results = ddgs.images(term, max_results=MAX_RESULTS_PER_TERM)
            except Exception as e:
                print(f"Rate limited or error searching {term}: {e}")
                time.sleep(10)
                continue

            if not results:
                continue

            for res in tqdm(results, desc=f"Downloading {term}"):
                image_url = res.get("image")
                if not image_url:
                    continue

                try:
                    response = requests.get(image_url, headers=headers, timeout=5)

                    if response.status_code == 200:
                        unique_id = str(uuid.uuid4())[:8]
                        file_ext = image_url.split(".")[-1].split("?")[0].lower()
                        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
                            file_ext = "jpg"

                        filename = PORK_DIR / f"scraped_pork_{unique_id}.{file_ext}"

                        with open(filename, "wb") as f:
                            f.write(response.content)
                        success_count += 1
                        time.sleep(0.5)  # Sleep to avoid rate limits
                except Exception:
                    pass

            # Sleep between search terms to avoid DuckDuckGo 403 Ratelimit
            print("Sleeping for 5 seconds to prevent rate limits...")
            time.sleep(5)

    print(f"\n=== Done! ===")
    print(f"Successfully scraped and downloaded {success_count} raw pork images into:")
    print(str(PORK_DIR))


if __name__ == "__main__":
    scrape_pork_meat()
