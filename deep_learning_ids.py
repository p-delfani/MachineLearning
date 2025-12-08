"""
Deep Learning Model for Intrusion Detection using PyTorch

Architecture:
- Fully connected neural network (MLP)
- Binary / Multi-class classification
"""

import torch
import torch.nn as nn
import torch.optim as optim

class IDSNet(nn.Module):
    """Neural network for intrusion detection."""
    def __init__(self, input_dim):
        super(IDSNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, dataloader, epochs=20, lr=1e-3):
    """Train the neural network."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
