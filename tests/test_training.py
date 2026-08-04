import numpy as np
import pytest
import torch

from train import TrackDataset


def test_track_dataset_loads_features_and_targets(tmp_path):
    """Check that saved NumPy arrays are loaded correctly."""
    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [5.0],
            [6.0],
        ],
        dtype=np.float32,
    )

    features_path = tmp_path / 'X.npy'
    targets_path = tmp_path / 'y.npy'

    np.save(features_path, features)
    np.save(targets_path, targets)

    dataset = TrackDataset(features_path, targets_path)

    assert len(dataset) == 2

    first_features, first_target = dataset[0]

    assert first_features.dtype == torch.float32
    assert first_target.dtype == torch.float32
    assert torch.allclose(
        first_features,
        torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(
        first_target,
        torch.tensor([5.0]),
    )


def test_track_dataset_rejects_mismatched_row_counts(tmp_path):
    """Check that features and targets must contain the same number of rows."""
    features_path = tmp_path / 'X.npy'
    targets_path = tmp_path / 'y.npy'

    np.save(
        features_path,
        np.zeros((3, 2), dtype=np.float32),
    )
    np.save(
        targets_path,
        np.zeros((2, 1), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match='same number of rows',
    ):
        TrackDataset(features_path, targets_path)


def test_track_dataset_reports_missing_files(tmp_path):
    """Check that missing preprocessing outputs produce a clear error."""
    with pytest.raises(
        FileNotFoundError,
        match='Run preprocessing.py first',
    ):
        TrackDataset(
            tmp_path / 'missing_X.npy',
            tmp_path / 'missing_y.npy',
        )