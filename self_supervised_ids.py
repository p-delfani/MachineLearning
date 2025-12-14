"""
Self-Supervised Learning for Intrusion Detection

Uses a pretext task to learn representations from unlabeled data,
then fine-tunes on labeled intrusion data.
"""

import torch
import torch.nn as nn

class SSL_IDS(nn.Module):
    """Self-supervised feature extractor."""
    def __init__(self, input_dim):
        super(SSL_IDS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.encoder(x)
