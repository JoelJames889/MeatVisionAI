import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import *
from dataset import DatasetLoader
from model import MeatVisionModel
from trainer import Trainer
from evaluate import Evaluator


def main():
    print("=" * 80)
    print("MEATVISION - FRESHNESS MODEL TRAINING")
    print("=" * 80)

    loader = DatasetLoader(FRESHNESS_DIR, batch_size=BATCH_SIZE, workers=NUM_WORKERS)
    train_loader, valid_loader, test_loader = loader.loaders()

    print("\nClasses:", loader.classes)
    print("Training Images :", len(loader.train_dataset))
    print("Validation Images :", len(loader.valid_dataset))
    print("Testing Images :", len(loader.test_dataset))

    model = MeatVisionModel(len(loader.classes)).get()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    trainer = Trainer(model, DEVICE, criterion, optimizer, scheduler)
    trainer.fit(train_loader, valid_loader, EPOCHS, MODEL_DIR / "freshness_model.pth")

    evaluator = Evaluator(model, DEVICE, loader.classes)
    evaluator.evaluate(test_loader)


if __name__ == "__main__":
    main()
