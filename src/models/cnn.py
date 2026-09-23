import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W each time

        # After 3 pools on a 224x224 input: 224 -> 112 -> 56 -> 28
        self.flattened_size = 64 * 28 * 28

        # Fully connected classifier head
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))   # -> [B, 64, 28, 28]

        x = x.view(x.size(0), -1)              # flatten to [B, 64*28*28]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # raw logits, shape [B, num_classes]

        return x