from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # Paths
    data_dir: str = ROOT / "data" / "raw" / "knee_images"
    csv_path: str = ROOT / "data" / "raw" / "images_image_split.csv"
    model_path: str = ROOT / "src" / "models" / "best_model.pt"
    log_path: str = ROOT / "src/models/training_log.csv"
    history_path: str = ROOT / "src/models/history.csv"
    class_folders: tuple = ("0", "1", "2", "3", "4")

    split_by: str = "image" # "image" (copies paper) or "patient" (prevents leakage)

    # Split
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    seed: int = 28

    # data
    num_classes: int = 5
    image_size: int = 224
    batch_size: int = 28
    num_workers: int = 4

    # training
    epochs: int = 100
    learning_rate: float = 1e-4
    patience: int = 5 # stop if val accuracy has not imporved after this many