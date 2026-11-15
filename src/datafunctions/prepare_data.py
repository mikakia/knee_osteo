# Partitions the data according to patient
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config

cfg = Config()
NAME = re.compile(r"^(\d+)[_\- ]?([LR])$", re.IGNORECASE)
data_dir = Path(cfg.data_dir)

# List every image
rows = []

for name in cfg.class_folders:
    class_folder = Path(cfg.data_dir) / name
    if not class_folder.is_dir():
        continue
    for image in sorted(class_folder.iterdir()):
        match = NAME.match(image.stem)
        if match is None:
            raise ValueError(f"Can't read patient number from: {image}")
        rows.append({
            "path": image.relative_to(data_dir).as_posix(),
            "label": int(class_folder.name),
            "patient": match.group(1),
            "side": match.group(2).upper(),
        })

table = pd.DataFrame(rows)


# Data checks
#print("Images per class:\n", table["label"].value_counts().sort_index().to_string())
#spanning = (table.groupby("patient")["label"].nunique() > 1).sum()
#print("Patients with images in more than one class:", spanning)

# Split by patient, keeping class balance as close as possible
def hold_out(df, fraction):
    """Set aside roughly 'fraction' of the patients"""
    n_folds = round(1/fraction)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    keep, held = next(splitter.split(df, df["label"], groups=df["patient"]))
    return df.iloc[keep], df.iloc[held]

if cfg.split_by == "image":
    train_val, test = train_test_split(
        table, test_size=cfg.test_fraction, stratify=table["label"], random_state=cfg.seed)
    train, val = train_test_split(
        train_val, test_size=cfg.val_fraction / (1 - cfg.test_fraction),
        stratify=train_val["label"], random_state=cfg.seed)
else:
    train_val, test = hold_out(table, cfg.test_fraction)
    train, val = hold_out(train_val, cfg.val_fraction / (1 - cfg.test_fraction))

table["split"] = "train"
table.loc[val.index, "split"] = "val"
table.loc[test.index, "split"] = "test"

if cfg.split_by == "patient":
    assert (table.groupby("patient")["split"].nunique() == 1).all(), "Patient leakage!"
else:
    shared = (table.groupby("patient")["split"].nunique() > 1).sum()
    print(f"Patients appearing in more than one split: {shared}") 

table.to_csv(cfg.csv_path, index=False)
print("\nImages per split and class:")
print(pd.crosstab(table["split"], table["label"]))