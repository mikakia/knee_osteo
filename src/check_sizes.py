import pandas as pd
from PIL import Image
from pathlib import Path
from config import Config

cfg = Config
table = pd.read_csv(cfg.csv_path)

sizes, modes = [], []
for path in table["path"]:
    with Image.open(Path(cfg.data_dir) / path) as img:
        sizes.append(img.size)
        modes.append(img.mode)

table["width"] = [s[0] for s in sizes]
table["height"] = [s[1] for s in sizes]
table["mode"] = modes

print(table[["width", "height"]].describe())
print("\nMost common sizes:")
print(table.groupby(["width", "height"]).size().sort_values(ascending=False).head(10))
print("\nColour modes:")
print(table["mode"].value_counts())


