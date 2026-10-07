import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from datafunctions.dataset import KneeOsteoDataset
from models.cnn import CNN


