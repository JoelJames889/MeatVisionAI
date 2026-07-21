import os
import requests
import uuid
from pathlib import Path
from duckduckgo_search import DDGS
from tqdm import tqdm

FISH_DIR = Path(r"d:\MeatVision_Project\Clean_Dataset\Species\fish")

# The search terms to grab exactly what the model is missing
SEARCH_TERMS = [
    "raw salmon fillet",
    "raw tuna steak",
    "raw white fish fillet meat",
    "raw cod fillet isolated",
    "raw fish meat isolated background",
]

MAX_RESULTS_PER_TERM = 100


def scrape_fish_meat():
    print(f"=== Phase 5c: Scraping Fish Meat Images ===")
    FISH_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    success_count = 0
    with DDGS() as ddgs:
        for term in SEARCH_TERMS:
            print(f"\nSearching for: '{term}'...")

            # Using duckduckgo image search
            results = ddgs.images(term, max_results=MAX_RESULTS_PER_TERM)

            if not results:
                print("No results found.")
                continue

            for res in tqdm(results, desc=f"Downloading {term}"):
                image_url = res.get("image")
                if not image_url:
                    continue

                try:
                    # Download the image with a timeout so it doesn't hang
                    response = requests.get(image_url, headers=headers, timeout=5)

                    if response.status_code == 200:
                        # Generate a unique ID so we don't overwrite anything
                        unique_id = str(uuid.uuid4())[:8]
                        file_ext = image_url.split(".")[-1].split("?")[0].lower()
                        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
                            file_ext = "jpg"

                        filename = (
                            FISH_DIR
                            / f"scraped_{term.replace(' ', '_')}_{unique_id}.{file_ext}"
                        )

                        with open(filename, "wb") as f:
                            f.write(response.content)
                        success_count += 1
                except Exception:
                    pass

    print(f"\n=== Done! ===")
    print(
        f"Successfully scraped and downloaded {success_count} raw fish meat images into:"
    )
    print(str(FISH_DIR))


if __name__ == "__main__":
    scrape_fish_meat()
