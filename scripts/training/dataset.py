import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from utils import get_train_transform, get_test_transform
from config import validate_dataset_exists


class TransformSubset(Dataset):
    """Zero-copy wrapper around a Subset or Dataset applying a specific transformation."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


class DatasetLoader:

    def __init__(self, data_dir, batch_size=32, workers=2, task: str = "species"):
        self.batch_size = batch_size
        self.workers = workers
        self.task = task

        validate_dataset_exists(data_dir, dataset_name=getattr(data_dir, "name", task))
        # Load dataset without transform on base ImageFolder so raw PIL is passed to TransformSubset
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

        train_targets = [full_dataset.targets[i] for i in train_ds.indices]

        # Apply specific transformations via lightweight zero-copy wrapper
        self.train_dataset = TransformSubset(
            train_ds, transform=get_train_transform(task=self.task)
        )
        self.train_dataset.targets = train_targets

        self.valid_dataset = TransformSubset(
            valid_ds, transform=get_test_transform()
        )
        self.test_dataset = TransformSubset(
            test_ds, transform=get_test_transform()
        )

        # Compute balanced sampler weights to handle class imbalance
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[target] for target in train_targets]
        self.sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
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
