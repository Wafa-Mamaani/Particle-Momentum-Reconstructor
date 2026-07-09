import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import TrackMomentumRegressor


class TrackDataset(Dataset):
    def __init__(self, features_path, targets_path):
        self.X = torch.tensor(np.load(features_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(targets_path), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(data_dir="data_files/processed_data", weights_dir="weights", epochs=50, batch_size=64):
    os.makedirs(weights_dir, exist_ok=True)

    train_dataset = TrackDataset(
        os.path.join(data_dir, "X_train.npy"),
        os.path.join(data_dir, "y_train.npy")
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TrackMomentumRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1} | Train loss: {train_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(weights_dir, "model.pth"))


if __name__ == "__main__":
    train_model()
