import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    A PyTorch Dataset for loading the preprocessed tracking arrays.
    By decoupling the data fetching logic from the training loop, we ensure that the DataLoader can efficiently batch, shuffle, and pin memory without cluttering the mathematical execution of the neural network.
    '''
    def __init__(self, features_path: str, targets_path: str):
        try:
            #Load numpy arrays and immediately convert them to 32-bit floats.
            self.X = torch.tensor(np.load(features_path), dtype = torch.float32)
            self.y = torch.tensor(np.load(targets_path), dtype = torch.float32)
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
    Executes the manual gradient descent and validation loops.
    Includes explicit early stopping and state saving.
    '''
    set_random_seed(seed)
    
    os.makedirs(weights_dir, exist_ok = True)

    #1. Hardware Selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training initialized on device: {device}')

    #2. Data Preparation
    train_dataset = TrackDataset(os.path.join(data_dir, 'X_train.npy'), os.path.join(data_dir, 'y_train.npy'))
    val_dataset = TrackDataset(os.path.join(data_dir, 'X_val.npy'), os.path.join(data_dir, 'y_val.npy'))

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

    #3. Model, Loss, and Optimizer Initialization
    model = TrackMomentumRegressor(input_dim = 72, pad_val = -9999.0).to(device)

    #Mean Squared Error
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    #4. The Epoch Loop
    for epoch in range(1, epochs + 1):
        #--- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        #tqdm wraps the loader to provide a clean CLI progress bar
        train_bar = tqdm(train_loader, desc = f'Epoch {epoch} / {epochs} [Train]', leave = False)

        for batch_X, batch_y in train_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            #Step: Zero the gradients from the previous batch
            optimizer.zero_grad()
            #Step: Forward Pass
            predictions = model(batch_X)
            #Step: Calculate the loss vector
            loss = criterion(predictions, batch_y)
            #Step: Backward Pass (Calculates dL/dW for all weights)
            loss.backward()
            #Step: Optimizer Step (Updates weights: W = W - lr * dL/dW)
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        #--- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch:03d} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}')

        #--- EARLY STOPPING & SAVING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            #Saving the raw weight matrices rather than the model object.
            save_path = os.path.join(weights_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered. Validation loss hasn\'t improved in {patience} epochs.')
                print(f'Best model weights saved to {save_path}')
                break
                

def main():
    '''Command line interface for the PyTorch training loop.'''
    parser = argparse.ArgumentParser(description = 'Train the MLP track momentum regressor.')
    parser.add_argument('--data', type = str, default = 'data_files/processed_data', help = 'Directory containing the .npy splits.')
    parser.add_argument('--weights', type = str, default = 'weights', help = 'Directory to save the trained model.')
    parser.add_argument('--epochs', type = int, default = 150, help = 'Maximum number of training epochs.')
    parser.add_argument('--batch', type = int, default = 64, help = 'Batch size for the DataLoader.')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate for the Adam optimizer.')
    parser.add_argument('--patience', type = int, default = 15, help = 'Epochs to wait for val loss improvement before stopping.')
    parser.add_argument(
        '--seed',
        type=int,
        default=13,
        help='Random seed for reproducible model training.',
    )   

    args = parser.parse_args()

    try:
        train_model(
            data_dir = args.data,
            weights_dir = args.weights,
            epochs = args.epochs,
            batch_size = args.batch,
            lr = args.lr,
            patience=args.patience,
            seed=args.seed,
        )
    except Exception as e:
        print(f'Training failed: {e}')
        

if __name__ == '__main__':
    main()

#Property of Wafa Mamani. May 2026.
