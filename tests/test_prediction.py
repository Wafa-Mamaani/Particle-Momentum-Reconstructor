import numpy as np
import pandas as pd
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