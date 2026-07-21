import os
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

CLEAN_DIR = Path(r"d:\MeatVision_Project\Clean_Dataset")
REPORTS_DIR = Path(r"d:\MeatVision_Project\Reports")


def run_ai_cleaner():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "phase5b_ai_cleaner_report.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenAI CLIP model on {device}...")

    # Load CLIP model and processor
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Define text prompts
    # If the image matches any of the "junk" categories higher than "raw meat", we delete it.
    categories = [
        "a photo of raw meat or raw fish",
        "a photo of an object, room, person, or scenery",
        "a photo of cooked meat or prepared food",
        "a photo of a lunchbox, container, or toy",
        "a digital illustration or cartoon",
    ]

    # We only care about index 0 being the highest
    valid_idx = 0

    image_paths = (
        list(CLEAN_DIR.rglob("*.jpg"))
        + list(CLEAN_DIR.rglob("*.png"))
        + list(CLEAN_DIR.rglob("*.jpeg"))
    )
    print(f"Found {len(image_paths)} images in Clean_Dataset to analyze.")

    deleted_count = 0

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== AI Junk Filtering Report ===\n")
        f.write(f"Total Scanned: {len(image_paths)}\n\n")

        for img_path in tqdm(image_paths, desc="AI Inspecting"):
            try:
                image = Image.open(img_path).convert("RGB")

                # Preprocess and run model
                inputs = processor(
                    text=categories, images=image, return_tensors="pt", padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Get probabilities
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).squeeze()

                best_idx = probs.argmax().item()

                # If the AI thinks it's NOT raw meat/fish
                if best_idx != valid_idx:
                    predicted_junk = categories[best_idx]
                    confidence = probs[best_idx].item() * 100

                    # Delete the image
                    img_path.unlink()
                    f.write(
                        f"DELETED: {img_path.name} - Detected as '{predicted_junk}' ({confidence:.1f}%)\n"
                    )
                    deleted_count += 1
            except Exception as e:
                pass

    print(f"\n=== AI Cleaning Complete ===")
    print(f"Deleted {deleted_count} completely irrelevant or junk images.")
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    print("=== Phase 5b: AI Junk Filter ===")
    run_ai_cleaner()
