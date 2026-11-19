import torch
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
import csv
from datetime import datetime
import time

class Trainer:
    def __init__(self, model, loaders, optimizer, cfg, device, scheduler=None):
        self.model = model.to(device)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

    def log(self, message):
        print(message)
        with open(self.cfg.log_path, "a") as f: # "a" = append, keeps old lines
            f.write(message + "\n")

    def run_epoch(self, split):
        training = (split == "train")
        self.model.train(training)
        if training and self.cfg.freeze_backbone:
            self.model.features.eval() # Keeps the batchnorm statistics fixed in the backbone since it's frozen
        total_loss, abs_error, correct, count = 0.0, 0, 0, 0

        with torch.set_grad_enabled(training):
            for images, labels in self.loaders[split]:
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images) # (batch, 4)
                loss = corn_loss(logits, labels, num_classes=self.cfg.num_classes)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                predicted = corn_label_from_logits(logits) # 0 to 4
                total_loss += loss.item() * len(labels)
                abs_error += (predicted - labels).abs().sum().item()
                correct += (predicted == labels).sum().item()
                count += len(labels)

        return total_loss / count, correct / count, abs_error / count

    def fit(self):
        best_mae, epochs_without_improvement = float("inf"), 0
        self.log(f"\n=== Run started {datetime.now():%Y-%m-%d %H:%M} ===")
        self.log(f"Settings: {self.cfg}")

        with open(self.cfg.history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "train_mae", "val_loss", "val_acc", "val_mae"])

            for epoch in range(1, self.cfg.epochs + 1):
                start = time.time()
                tr_loss, tr_acc, tr_mae = self.run_epoch("train")
                va_loss, va_acc, va_mae = self.run_epoch("val")
                minutes = (time.time() - start) / 60

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(va_mae)
                    else:
                        self.scheduler.step()          

                lr = self.optimizer.param_groups[0]["lr"]

                self.log(f"Epoch {epoch:2d} | lr {lr:.2e} | train loss {tr_loss:.3f} MAE {tr_mae:.3f} "
                         f"| val loss {va_loss:.3f} acc {va_acc:.3f} MAE {va_mae:.3f} | {minutes:.1f} min")
                writer.writerow([epoch, tr_loss, tr_acc, tr_mae, va_loss, va_acc, va_mae])
                f.flush() # write to disk now, not at the end

                if va_mae < best_mae:
                    best_mae = va_mae
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), self.cfg.model_path)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.cfg.patience:
                        self.log("Validation MAE stopped imporving, stopping early.")
                        break

            self.log(f"Best validation MAE: {best_mae:.3f}")