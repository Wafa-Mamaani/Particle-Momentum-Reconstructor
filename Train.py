import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import TrackMomentumRegressor


def train_model(data_dir="data_files/processed_data", weights_dir="weights", epochs=50):
    os.makedirs(weights_dir, exist_ok=True)

    X_train = torch.tensor(np.load(os.path.join(data_dir, "X_train.npy")), dtype=torch.float32)
    y_train = torch.tensor(np.load(os.path.join(data_dir, "y_train.npy")), dtype=torch.float32)

    model = TrackMomentumRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        print("epoch", epoch + 1, "loss", loss.item())

    torch.save(model.state_dict(), os.path.join(weights_dir, "model.pth"))


if __name__ == "__main__":
    train_model()
