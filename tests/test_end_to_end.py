import numpy as np
import pandas as pd
import torch

from plot import plot_results
from predict import run_inference
from preprocessing import TrackPreprocessor
from simulation import generate_toy_tracks
from train import train_model


def test_complete_reconstruction_pipeline(tmp_path, monkeypatch):
    """Check that the full reconstruction workflow runs end to end."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    simulated_dir = tmp_path / 'simulated_data'
    processed_dir = tmp_path / 'processed_data'
    weights_dir = tmp_path / 'weights'
    results_dir = tmp_path / 'results'
    plots_dir = results_dir / 'plots'

    simulated_dir.mkdir()

    # 1. Simulation
    tracks = generate_toy_tracks(
        num_samples=40,
        random_seed=13,
    )

    csv_path = simulated_dir / 'simulated_tracks.csv'
    tracks.to_csv(csv_path, index=False)

    assert csv_path.exists()

    # 2. Preprocessing
    processor = TrackPreprocessor(
        filepath=str(csv_path),
        random_state=13,
    )

    processor.run_pipeline(str(processed_dir))

    assert (processed_dir / 'X_train.npy').exists()
    assert (processed_dir / 'X_val.npy').exists()
    assert (processed_dir / 'X_test.npy').exists()
    assert (processed_dir / 'y_stats.npz').exists()

    # 3. Training
    train_model(
        data_dir=str(processed_dir),
        weights_dir=str(weights_dir),
        epochs=1,
        batch_size=8,
        lr=0.001,
        patience=1,
        seed=13,
    )

    weights_path = weights_dir / 'best_model.pth'

    assert weights_path.exists()

    # 4. Prediction
    run_inference(
        data_dir=str(processed_dir),
        weights_path=str(weights_path),
        output_dir=str(results_dir),
    )

    predictions_path = results_dir / 'test_predictions.csv'

    assert predictions_path.exists()

    results = pd.read_csv(predictions_path)
    X_test = np.load(processed_dir / 'X_test.npy')

    assert list(results.columns) == [
        'pt_true_mev',
        'pt_pred_mev',
    ]
    assert len(results) == X_test.shape[0]
    assert np.all(
        np.isfinite(
            results[['pt_true_mev', 'pt_pred_mev']].to_numpy()
        )
    )

    # 5. Evaluation plots
    plot_results(
        csv_path=str(predictions_path),
        output_dir=str(plots_dir),
    )

    scatter_path = plots_dir / 'reconstruction_scatter.png'
    residuals_path = plots_dir / 'error_residuals.png'

    assert scatter_path.exists()
    assert residuals_path.exists()

    assert scatter_path.stat().st_size > 0
    assert residuals_path.stat().st_size > 0
