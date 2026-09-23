import os
from collections import Counter

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class KneeOsteoDataset(Dataset):
    
    """Wraps torchvision.ImageFolder for the knee osteoarthritis dataset."""

    def __init__(self, root_dir=None, transform=None):
        if root_dir is None:
            # Resolve path relative to project root, not hardcoded to your machine
            root_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "raw", "knee_images"
            )
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.dataset = datasets.ImageFolder(root=root_dir, transform=self.transform)

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def class_counts(self):
        counts = Counter(self.targets)
        return {self.classes[i]: counts[i] for i in sorted(counts)}

    def summary(self):
        print(f"Classes: {self.classes}")
        print(f"Class to index: {self.class_to_idx}")
        print(f"Total images: {len(self)}")
        for cls, count in self.class_counts().items():
            print(f"Class {cls}: {count} images")