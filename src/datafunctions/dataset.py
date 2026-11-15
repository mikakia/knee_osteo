import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class KneeOsteoDataset(Dataset):
    """One (image,label) pair at a time"""

    def __init__(self, table, transform, data_dir):
        self.table = table.reset_index(drop=True)
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        image = Image.open(self.data_dir / row["path"]).convert("RGB") # Converting to 3 color channels as that is what DenseNet expects
        return self.transform(image), int(row["label"])

def get_loaders(cfg):
    size = (cfg.image_size, cfg.image_size)

    # Image transforms mentioned in paper (brightness, saturation, flip horizontal, random affine, normalize)
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ColorJitter(brightness=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=15, # Rotation in [-15, 15]
            translate=(0.1,0.1), # Up to 10% shift in x and y
            scale=(0.9, 1.1), # Zoom in/out
            shear=10, # Shear in [-10, 10]
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD) # Normalized according to ImageNet
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    table = pd.read_csv(cfg.csv_path, dtype={"patient": str})
    loaders = {}
    for name, tf in [("train", train_tf), ("val", eval_tf), ("test", eval_tf)]:
        data = KneeOsteoDataset(table[table["split"] == name], tf, cfg.data_dir)
        loaders[name] = DataLoader(data, batch_size=cfg.batch_size,
                                   shuffle=(name == "train"),
                                   num_workers=cfg.num_workers)
    return loaders