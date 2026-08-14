import numpy as np
import pytest
import torch

from model import TrackMomentumRegressor
from train import TrackDataset, train_model


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


def test_train_model_creates_valid_checkpoint(tmp_path):
    """Check that a short training run creates loadable model weights."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    X_train = rng.normal(size=(12, 72)).astype(np.float32)
    y_train = rng.normal(size=(12, 1)).astype(np.float32)
    X_val = rng.normal(size=(4, 72)).astype(np.float32)
    y_val = rng.normal(size=(4, 1)).astype(np.float32)

    np.save(data_dir / 'X_train.npy', X_train)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'X_val.npy', X_val)
    np.save(data_dir / 'y_val.npy', y_val)

    train_model(
        data_dir=str(data_dir),
        weights_dir=str(weights_dir),
        epochs=1,
        batch_size=4,
        lr=0.001,
        patience=1,
        seed=13,
    )

    checkpoint_path = weights_dir / 'best_model.pth'

    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0

    state_dict = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )

    model = TrackMomentumRegressor(
        input_dim=72,
        pad_val=-9999.0,
    )
    model.load_state_dict(state_dict)


def test_train_model_is_reproducible_with_same_seed(
    tmp_path,
    monkeypatch,
):
    """Check that identical seeds produce identical trained weights."""
    # Force CPU execution so the test behaves consistently on every machine.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    data_dir = tmp_path / 'processed_data'
    first_weights_dir = tmp_path / 'weights_run_1'
    second_weights_dir = tmp_path / 'weights_run_2'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    X_train = rng.normal(size=(12, 72)).astype(np.float32)
    y_train = rng.normal(size=(12, 1)).astype(np.float32)
    X_val = rng.normal(size=(4, 72)).astype(np.float32)
    y_val = rng.normal(size=(4, 1)).astype(np.float32)

    np.save(data_dir / 'X_train.npy', X_train)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'X_val.npy', X_val)
    np.save(data_dir / 'y_val.npy', y_val)

    common_arguments = {
        'data_dir': str(data_dir),
        'epochs': 1,
        'batch_size': 4,
        'lr': 0.001,
        'patience': 1,
        'seed': 13,
    }

    train_model(
        weights_dir=str(first_weights_dir),
        **common_arguments,
    )
    train_model(
        weights_dir=str(second_weights_dir),
        **common_arguments,
    )

    first_state_dict = torch.load(
        first_weights_dir / 'best_model.pth',
        map_location='cpu',
        weights_only=True,
    )
    second_state_dict = torch.load(
        second_weights_dir / 'best_model.pth',
        map_location='cpu',
        weights_only=True,
    )

    assert list(first_state_dict) == list(second_state_dict)

    for parameter_name in first_state_dict:
        assert torch.equal(
            first_state_dict[parameter_name],
            second_state_dict[parameter_name],
        ), f'Different values found in {parameter_name}'


def test_train_model_stops_early_without_validation_improvement(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Check that training stops when validation loss stops improving."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    # Prevent optimizer updates so validation loss remains unchanged.
    monkeypatch.setattr(
        torch.optim.Adam,
        'step',
        lambda self: None,
    )

    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    X_train = rng.normal(size=(12, 72)).astype(np.float32)
    y_train = rng.normal(size=(12, 1)).astype(np.float32)
    X_val = rng.normal(size=(4, 72)).astype(np.float32)
    y_val = rng.normal(size=(4, 1)).astype(np.float32)

    np.save(data_dir / 'X_train.npy', X_train)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'X_val.npy', X_val)
    np.save(data_dir / 'y_val.npy', y_val)

    train_model(
        data_dir=str(data_dir),
        weights_dir=str(weights_dir),
        epochs=5,
        batch_size=4,
        lr=0.001,
        patience=1,
        seed=13,
    )

    output = capsys.readouterr().out

    assert 'Early stopping triggered' in output
    assert 'Epoch 002' in output
    assert 'Epoch 003' not in output
    assert (weights_dir / 'best_model.pth').exists()


def test_train_model_rejects_non_positive_epochs(tmp_path):
    """Check that the number of training epochs must be positive."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    np.save(
        data_dir / 'X_train.npy',
        rng.normal(size=(12, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_train.npy',
        rng.normal(size=(12, 1)).astype(np.float32),
    )
    np.save(
        data_dir / 'X_val.npy',
        rng.normal(size=(4, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_val.npy',
        rng.normal(size=(4, 1)).astype(np.float32),
    )

    with pytest.raises(
        ValueError,
        match='epochs must be positive',
    ):
        train_model(
            data_dir=str(data_dir),
            weights_dir=str(weights_dir),
            epochs=0,
            batch_size=4,
            lr=0.001,
            patience=2,
            seed=13,
        )


def test_train_model_rejects_non_positive_batch_size(tmp_path):
    """Check that batch size must be positive."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    np.save(
        data_dir / 'X_train.npy',
        rng.normal(size=(12, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_train.npy',
        rng.normal(size=(12, 1)).astype(np.float32),
    )
    np.save(
        data_dir / 'X_val.npy',
        rng.normal(size=(4, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_val.npy',
        rng.normal(size=(4, 1)).astype(np.float32),
    )

    with pytest.raises(
        ValueError,
        match='batch_size must be positive',
    ):
        train_model(
            data_dir=str(data_dir),
            weights_dir=str(weights_dir),
            epochs=1,
            batch_size=0,
            lr=0.001,
            patience=2,
            seed=13,
        )


def test_train_model_rejects_non_positive_learning_rate(tmp_path):
    """Check that learning rate must be positive."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    np.save(
        data_dir / 'X_train.npy',
        rng.normal(size=(12, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_train.npy',
        rng.normal(size=(12, 1)).astype(np.float32),
    )
    np.save(
        data_dir / 'X_val.npy',
        rng.normal(size=(4, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_val.npy',
        rng.normal(size=(4, 1)).astype(np.float32),
    )

    with pytest.raises(
        ValueError,
        match='lr must be positive',
    ):
        train_model(
            data_dir=str(data_dir),
            weights_dir=str(weights_dir),
            epochs=1,
            batch_size=4,
            lr=0.0,
            patience=2,
            seed=13,
        )


def test_train_model_rejects_non_positive_patience(tmp_path):
    """Check that early-stopping patience must be positive."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    np.save(
        data_dir / 'X_train.npy',
        rng.normal(size=(12, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_train.npy',
        rng.normal(size=(12, 1)).astype(np.float32),
    )
    np.save(
        data_dir / 'X_val.npy',
        rng.normal(size=(4, 72)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_val.npy',
        rng.normal(size=(4, 1)).astype(np.float32),
    )

    with pytest.raises(
        ValueError,
        match='patience must be positive',
    ):
        train_model(
            data_dir=str(data_dir),
            weights_dir=str(weights_dir),
            epochs=1,
            batch_size=4,
            lr=0.001,
            patience=0,
            seed=13,
        )


def test_train_model_infers_input_dimension_from_data(
    tmp_path,
    monkeypatch,
):
    """Check that model input size is inferred from processed features."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    # Use 8 features instead of the project's usual 72.
    np.save(
        data_dir / 'X_train.npy',
        rng.normal(size=(12, 8)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_train.npy',
        rng.normal(size=(12, 1)).astype(np.float32),
    )
    np.save(
        data_dir / 'X_val.npy',
        rng.normal(size=(4, 8)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_val.npy',
        rng.normal(size=(4, 1)).astype(np.float32),
    )

    train_model(
        data_dir=str(data_dir),
        weights_dir=str(weights_dir),
        epochs=1,
        batch_size=4,
        lr=0.001,
        patience=2,
        seed=13,
    )

    assert (weights_dir / 'best_model.pth').exists()


def test_train_model_rejects_mismatched_feature_dimensions(tmp_path):
    """Check that training and validation feature dimensions match."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    data_dir.mkdir()

    rng = np.random.default_rng(13)

    np.save(
        data_dir / 'X_train.npy',
        rng.normal(size=(12, 8)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_train.npy',
        rng.normal(size=(12, 1)).astype(np.float32),
    )

    # Deliberately use a different feature dimension.
    np.save(
        data_dir / 'X_val.npy',
        rng.normal(size=(4, 10)).astype(np.float32),
    )
    np.save(
        data_dir / 'y_val.npy',
        rng.normal(size=(4, 1)).astype(np.float32),
    )

    with pytest.raises(
        ValueError,
        match='Training and validation feature dimensions must match',
    ):
        train_model(
            data_dir=str(data_dir),
            weights_dir=str(weights_dir),
            epochs=1,
            batch_size=4,
            lr=0.001,
            patience=2,
            seed=13,
        )