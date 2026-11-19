from torch import nn
from torchvision import models

def build_model(num_classes, freeze_backbone=False):
    model = models.densenet161(weights="DEFAULT") # Pretrained on ImageNet

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False # Freeze everything

    # Create new head trainable by default
    model.classifier = nn.Linear(model.classifier.in_features, num_classes - 1) # Outputs a CORN layer
    return model
