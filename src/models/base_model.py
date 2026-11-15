from torch import nn
from torchvision import models

def build_model(num_classes):
    model = models.densenet161(weights="DEFAULT") # Pretrained on ImageNet
    model.classifier = nn.Linear(model.classifier.in_features, num_classes - 1)
    return model