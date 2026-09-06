import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Set up paths relative to the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import DatasetLoader
from model import MeatVisionModel
from evaluate import Evaluator
import config

def evaluate_model():
    print("Evaluating Freshness Model...")
    # Override dataset path
    config.DATASET = config.PROJECT_ROOT / "Dataset"
    config.FRESHNESS_DIR = config.DATASET / "freshness"

    print(f"Using Dataset at: {config.FRESHNESS_DIR}")

    # Initialize DatasetLoader
    loader = DatasetLoader(config.FRESHNESS_DIR, batch_size=64, workers=0) 
    _, _, test_loader = loader.loaders()

    print(f"Classes: {loader.classes}")
    print(f"Testing Images : {len(loader.test_dataset)}")

    # Initialize Model
    model = MeatVisionModel(len(loader.classes)).get()
    model_path = config.MODEL_DIR / "freshness_model.pth"
    print(f"Loading weights from {model_path}")
    
    if not model_path.exists():
        print("Model file does not exist!")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)

    # Evaluate
    evaluator = Evaluator(model, config.DEVICE, loader.classes)
    metrics = evaluator.evaluate(test_loader)

    print("\nFINAL FRESHNESS METRICS:")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score : {metrics['f1'] * 100:.2f}%")

if __name__ == '__main__':
    evaluate_model()
