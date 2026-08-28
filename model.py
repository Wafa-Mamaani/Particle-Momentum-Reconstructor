import torch
import torch.nn as nn


class TrackMomentumRegressor(nn.Module):
    '''
    A Multi-Layer Perceptron (MLP) for predicting particle transverse
    momentum from 2D tracker hit coordinates.

    Missing hits marked by the padding value are replaced with zero before
    the input is passed through the network.
    '''

    def __init__(self, input_dim: int = 72, pad_val: float = -9999.0):
        """
        Initializes the network layers.

        Parameters
        ----------
        input_dim : int
            Number of tracker-coordinate features given to the network.
            The default detector layout produces 72 features.
        pad_val : float
            The sentinel value used to denote a missing or padded hit.
        """
        super().__init__()
        self.pad_val = pad_val

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            A batch of track features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted scaled pT values of shape (batch_size, 1).
        """
        mask = (x != self.pad_val).float()
        masked_x = x * mask

        return self.network(masked_x)


# Property of Wafa Mamaani. May 2026.
