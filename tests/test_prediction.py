import numpy as np
import pandas as pd
import pytest
import torch

from model import TrackMomentumRegressor
from predict import run_inference


def test_run_inference_restores_physical_units(
    tmp_path,
    monkeypatch,
):
    """Check prediction output and inverse target scaling."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    output_dir = tmp_path / 'results'

    data_dir.mkdir()
    weights_dir.mkdir()

    X_test = np.zeros((3, 72), dtype=np.float32)
    y_test_scaled = np.array(
        [
            [-1.0],
            [0.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    y_mean = np.array(85.0, dtype=np.float32)
    y_std = np.array(2.0, dtype=np.float32)

    np.save(data_dir / 'X_test.npy', X_test)
    np.save(data_dir / 'y_test.npy', y_test_scaled)
    np.savez(
        data_dir / 'y_stats.npz',
        y_mean=y_mean,
        y_std=y_std,
    )

    model = TrackMomentumRegressor(
        input_dim=72,
        pad_val=-9999.0,
    )

    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    weights_path = weights_dir / 'best_model.pth'
    torch.save(model.state_dict(), weights_path)

    run_inference(
        data_dir=str(data_dir),
        weights_path=str(weights_path),
        output_dir=str(output_dir),
    )

    predictions_path = output_dir / 'test_predictions.csv'

    assert predictions_path.exists()

    results = pd.read_csv(predictions_path)

    assert list(results.columns) == [
        'pt_true_mev',
        'pt_pred_mev',
    ]

    np.testing.assert_allclose(
        results['pt_true_mev'].to_numpy(),
        np.array([83.0, 85.0, 87.0]),
    )
    np.testing.assert_allclose(
        results['pt_pred_mev'].to_numpy(),
        np.array([85.0, 85.0, 85.0]),
    )


def test_run_inference_reports_missing_processed_data(
    tmp_path,
    monkeypatch,
):
    """Check that missing processed files produce a clear error."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    missing_data_dir = tmp_path / 'missing_processed_data'
    weights_path = tmp_path / 'missing_weights.pth'
    output_dir = tmp_path / 'results'

    with pytest.raises(
        FileNotFoundError,
        match='Run preprocessing first',
    ):
        run_inference(
            data_dir=str(missing_data_dir),
            weights_path=str(weights_path),
            output_dir=str(output_dir),
        )


def test_run_inference_reports_missing_model_weights(
    tmp_path,
    monkeypatch,
):
    """Check that a missing model checkpoint produces a clear error."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    data_dir = tmp_path / 'processed_data'
    output_dir = tmp_path / 'results'
    missing_weights_path = tmp_path / 'missing_model.pth'
    data_dir.mkdir()

    np.save(
        data_dir / 'X_test.npy',
        np.zeros((1, 72), dtype=np.float32),
    )
    np.save(
        data_dir / 'y_test.npy',
        np.zeros((1, 1), dtype=np.float32),
    )
    np.savez(
        data_dir / 'y_stats.npz',
        y_mean=np.array(85.0, dtype=np.float32),
        y_std=np.array(2.0, dtype=np.float32),
    )

    with pytest.raises(
        FileNotFoundError,
        match='Model weights not found',
    ):
        run_inference(
            data_dir=str(data_dir),
            weights_path=str(missing_weights_path),
            output_dir=str(output_dir),
        )


def test_prediction_infers_input_dimension_from_data(tmp_path):
    """Check that prediction works with a non-default feature dimension."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    output_dir = tmp_path / 'results'

    data_dir.mkdir()
    weights_dir.mkdir()

    X_test = np.zeros((3, 8), dtype=np.float32)
    y_test = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)

    np.save(data_dir / 'X_test.npy', X_test)
    np.save(data_dir / 'y_test.npy', y_test)

    np.savez(
        data_dir / 'y_stats.npz',
        y_mean=np.array([85.0]),
        y_std=np.array([2.0]),
    )

    model = TrackMomentumRegressor(
        input_dim=8,
        pad_val=-9999.0,
    )

    torch.save(
        model.state_dict(),
        weights_dir / 'best_model.pth',
    )

    run_inference(
        data_dir=str(data_dir),
        weights_path=str(weights_dir / 'best_model.pth'),
        output_dir=str(output_dir),
    )

    assert (output_dir / 'test_predictions.csv').exists()


def test_run_inference_rejects_mismatched_row_counts(tmp_path):
    '''Test features and targets must contain the same number of tracks.'''

    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    output_dir = tmp_path / 'results'

    data_dir.mkdir()
    weights_dir.mkdir()

    np.save(
        data_dir / 'X_test.npy',
        np.zeros((3, 8), dtype=np.float32),
    )
    np.save(
        data_dir / 'y_test.npy',
        np.zeros((2, 1), dtype=np.float32),
    )
    np.savez(
        data_dir / 'y_stats.npz',
        y_mean=np.array([85.0]),
        y_std=np.array([2.0]),
    )

    model = TrackMomentumRegressor(
        input_dim=8,
        pad_val=-9999.0,
    )
    torch.save(
        model.state_dict(),
        weights_dir / 'best_model.pth',
    )

    with pytest.raises(
        ValueError,
        match='same number of rows',
    ):
        run_inference(
            data_dir=str(data_dir),
            weights_path=str(weights_dir / 'best_model.pth'),
            output_dir=str(output_dir),
        )


def test_run_inference_rejects_mismatched_feature_dimension(tmp_path):
    """Check that test features match the trained model input dimension."""
    data_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    output_dir = tmp_path / 'results'

    data_dir.mkdir()
    weights_dir.mkdir()

    # Test data has 10 features.
    np.save(
        data_dir / 'X_test.npy',
        np.zeros((3, 10), dtype=np.float32),
    )
    np.save(
        data_dir / 'y_test.npy',
        np.zeros((3, 1), dtype=np.float32),
    )

    np.savez(
        data_dir / 'y_stats.npz',
        y_mean=np.array([85.0]),
        y_std=np.array([2.0]),
    )

    # Trained model expects only 8 features.
    model = TrackMomentumRegressor(
        input_dim=8,
        pad_val=-9999.0,
    )

    torch.save(
        model.state_dict(),
        weights_dir / 'best_model.pth',
    )

    with pytest.raises(
        ValueError,
        match='Test feature dimension does not match the trained model',
    ):
        run_inference(
            data_dir=str(data_dir),
            weights_path=str(weights_dir / 'best_model.pth'),
            output_dir=str(output_dir),
        )
