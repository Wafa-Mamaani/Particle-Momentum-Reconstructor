import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import TrackMomentumRegressor


def set_random_seed(seed: int) -> None:
    """Set the random seeds used during model training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TrackDataset(Dataset):
    '''
    Loads the preprocessed feature and target arrays for PyTorch training.
    '''
    def __init__(self, features_path: str, targets_path: str):
        try:
            # Load the saved arrays as 32-bit floating-point tensors.
            self.X = torch.tensor(
                np.load(features_path),
                dtype=torch.float32,
            )
            self.y = torch.tensor(
                np.load(targets_path),
                dtype=torch.float32,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not locate {features_path} or {targets_path}. Run preprocessing.py first.')

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('Feature and target arrays must have the same number of rows.')

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


def train_model(
    data_dir: str,
    weights_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 10,
    seed: int = 13,
):
    '''
    Trains the momentum regressor with Adam and evaluates it on the
    validation set after each epoch. The best model weights are saved,
    with early stopping when validation loss stops improving.
    '''
    if epochs <= 0:
        raise ValueError('epochs must be positive')

    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    if lr <= 0:
        raise ValueError('lr must be positive')

    if patience <= 0:
        raise ValueError('patience must be positive')
    
    set_random_seed(seed)
    
    os.makedirs(weights_dir, exist_ok=True)

    # Select CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training initialized on device: {device}')

    # Load training and validation data
    train_dataset = TrackDataset(
        os.path.join(data_dir, 'X_train.npy'),
        os.path.join(data_dir, 'y_train.npy'),
    )

    val_dataset = TrackDataset(
        os.path.join(data_dir, 'X_val.npy'),
        os.path.join(data_dir, 'y_val.npy'),
    )

    if train_dataset.X.shape[1] != val_dataset.X.shape[1]:
        raise ValueError(
            'Training and validation feature dimensions must match.'
        )

    # The DataLoader handles batching and reproducible shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize the model, loss function, and optimizer
    input_dim = train_dataset.X.shape[1]

    model = TrackMomentumRegressor(
        input_dim=input_dim,
        pad_val=-9999.0,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch} / {epochs} [Train]',
            leave=False,
        )
        for batch_X, batch_y in train_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f'Epoch {epoch:03d} | '
            f'Train Loss (MSE): {train_loss:.4f} | '
            f'Val Loss (MSE): {val_loss:.4f}'
        )

        # Save the best validation result and apply early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save only the model state so the architecture remains defined in model.py.
            save_path = os.path.join(weights_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f'\nEarly stopping triggered. Validation loss has not improved '
                    f'in {patience} epochs.'
                )
                print(f'Best model weights saved to {save_path}')
                break
                

def main():  # pragma: no cover
    '''Command line interface for the PyTorch training loop.'''
    parser = argparse.ArgumentParser(
        description='Train the MLP track momentum regressor.'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data_files/processed_data',
        help='Directory containing the .npy splits.',
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='weights',
        help='Directory to save the trained model.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Maximum number of training epochs.',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=64,
        help='Batch size for the DataLoader.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate for the Adam optimizer.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Epochs to wait for val loss improvement before stopping.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=13,
        help='Random seed for reproducible model training.',
    ) 
      
    args = parser.parse_args()

    try:
        train_model(
            data_dir=args.data,
            weights_dir=args.weights,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'Training failed: {exc}') from exc


if __name__ == '__main__':  # pragma: no cover
    main()

# Property of Wafa Mamaani. May 2026.
