from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # Paths
    data_dir: str = ROOT / "data" / "raw" / "knee_images"       # Leads to parent folder of all classes of images
    csv_path: str = ROOT / "data" / "raw" / "images.csv"        # CSV of data splitting
    model_path: str = ROOT / "models" / "best_model.pt"         # Edit name or change other model names to not overwrite existing ones
    log_path: str = ROOT / "models" / "training_log.csv"        # Copy of the printouts, saves configuration at top of each run
    history_path: str = ROOT / "src" / "models" / "history.csv" # Edit name or change saved names to not overwrite, history for current model 
    class_folders: tuple = ("0", "1", "2", "3", "4")

    split_by: str = "patient" # "image" (just by image) or "patient" (prevents leakage)

    # Split
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    seed: int = 28

    # data
    num_classes: int = 5            
    image_size: int = 224   # Change depending on model input   
    batch_size: int = 28 
    num_workers: int = 4

    # training
    epochs: int = 100               # Total epochs training will run for
    learning_rate: float = 1e-4     # Initial learning rate
    patience: int = 100             # Stop training if no improvement past so many epochs
    freeze_backbone: bool = True    # Whether you want to freeze pre-trained model