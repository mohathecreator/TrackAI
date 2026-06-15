import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackmaniaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 80 * 45, 256)
        self.fc2 = nn.Linear(256, 4)

        self.dropout = nn.Dropout(0.5)


