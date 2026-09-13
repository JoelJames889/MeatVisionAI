import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from utils import get_train_transform, get_test_transform
from config import validate_dataset_exists
import copy


class DatasetLoader:

    def __init__(self, data_dir, batch_size=32, workers=2):
        self.batch_size = batch_size
        self.workers = workers

        validate_dataset_exists(data_dir, dataset_name=getattr(data_dir, "name", "species"))
        full_dataset = datasets.ImageFolder(data_dir)
        self.classes = full_dataset.classes

        # Calculate split sizes (80% train, 10% valid, 10% test)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        valid_size = int(0.1 * total_size)
        test_size = total_size - train_size - valid_size

        # Split the dataset randomly
        train_ds, valid_ds, test_ds = random_split(
            full_dataset,
            [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Apply specific transformations
        self.train_dataset = copy.deepcopy(train_ds)
        self.train_dataset.dataset.transform = get_train_transform()
        self.train_dataset.targets = [full_dataset.targets[i] for i in train_ds.indices]

        self.valid_dataset = copy.deepcopy(valid_ds)
        self.valid_dataset.dataset.transform = get_test_transform()

        self.test_dataset = copy.deepcopy(test_ds)
        self.test_dataset.dataset.transform = get_test_transform()

        # Compute balanced sampler weights to handle 10:1 class imbalance
        class_counts = np.bincount(self.train_dataset.targets)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[target] for target in self.train_dataset.targets]
        self.sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

    def loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.workers,
            drop_last=True,
        )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        return train_loader, valid_loader, test_loader
