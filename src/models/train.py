import random
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.datafunctions.dataset import get_loaders
from base_model import build_model
from trainer import Trainer

def main():
    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    loaders = get_loaders(cfg)
    model = build_model(cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # halve every 5 epochs

    Trainer(model, loaders, optimizer, cfg, device, scheduler=scheduler).fit()

if __name__ == "__main__":
    main()
