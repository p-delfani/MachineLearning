"""
LSTM-based Intrusion Detection System

This model captures temporal dependencies in network traffic,
which is critical for detecting sequential attack patterns.
"""

import torch
import torch.nn as nn

class LSTM_IDS(nn.Module):
    """LSTM model for intrusion detection."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LSTM_IDS, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.sigmoid(self.fc(last_output))
