import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from coral_pytorch.dataset import corn_label_from_logits
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.datafunctions.dataset import get_loaders
from base_model import build_model

def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = get_loaders(cfg)
    model = build_model(cfg.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval

    guesses, answers = [], []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            logits = model(images.to(device))
            guesses += corn_label_from_logits(logits).cpu().tolist()
            answers += labels.tolist()

    guesses, answers = np.array(guesses), np.array(answers)
    print(classification_report(answers, guesses, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(answers, guesses))
    print(f"MAE: {np.abs(guesses - answers).mean():.3f}")
    print(f"Quadratic weighted kappa: {cohen_kappa_score(answers, guesses, weights='quadratic'):.3f}")

if __name__ == "__main__":
    main()