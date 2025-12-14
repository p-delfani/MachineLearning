"""
Autoencoder-based Anomaly Detection for IDS

The model learns normal traffic patterns and detects intrusions
based on reconstruction error.
"""

import torch
import torch.nn as nn

class AutoencoderIDS(nn.Module):
    """Autoencoder for anomaly-based intrusion detection."""
    def __init__(self, input_dim):
        super(AutoencoderIDS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
