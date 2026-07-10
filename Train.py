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

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Feature and target arrays must have the same number of rows.")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(data_dir="data_files/processed_data", weights_dir="weights", epochs=100, batch_size=64, lr=0.001):
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_dataset = TrackDataset(os.path.join(data_dir, "X_train.npy"), os.path.join(data_dir, "y_train.npy"))
    val_dataset = TrackDataset(os.path.join(data_dir, "X_val.npy"), os.path.join(data_dir, "y_val.npy"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TrackMomentumRegressor(input_dim=72, pad_val=-9999.0).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(weights_dir, "model.pth"))


if __name__ == "__main__":
    train_model()
