import torch
import torch.nn as nn


class TrackMomentumModel(nn.Module):
    def __init__(self, input_dim=72, pad_val=-9999.0):
        super().__init__()
        self.pad_val = pad_val

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # First attempt to stop missing detector hits from dominating the model.
        # The preprocessing script keeps missing hits as -9999, so they should not
        # be passed directly into the dense layers.
        mask = (x != self.pad_val).float()
        x = x * mask

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
