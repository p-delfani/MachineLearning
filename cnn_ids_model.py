"""
CNN-based Intrusion Detection System

CNNs are used to automatically learn spatial feature patterns
from network traffic represented as matrices.
"""

import torch
import torch.nn as nn

class CNN_IDS(nn.Module):
    """Convolutional Neural Network for IDS."""
    def __init__(self, input_channels=1):
        super(CNN_IDS, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
