import numpy as np
import pandas as pd
import pytest


from plot import calculate_metrics, plot_results


def test_calculate_metrics_returns_expected_values():
    """Check residuals, bias, standard deviation, and RMSE."""
    y_true = np.array([80.0, 85.0, 90.0])
    y_pred = np.array([81.0, 83.0, 91.0])

    residuals, bias, standard_deviation, rmse = calculate_metrics(
        y_true,
        y_pred,
    )

    np.testing.assert_allclose(
        residuals,
        np.array([1.0, -2.0, 1.0]),
    )
    assert bias == pytest.approx(0.0)
    assert standard_deviation == pytest.approx(np.sqrt(3.0))
    assert rmse == pytest.approx(np.sqrt(2.0))


def test_calculate_metrics_rejects_different_shapes():
    """Check that true and predicted arrays must have matching shapes."""
    y_true = np.array([80.0, 85.0])
    y_pred = np.array([81.0])

    with pytest.raises(
        ValueError,
        match='same shape',
    ):
        calculate_metrics(y_true, y_pred)


def test_calculate_metrics_requires_at_least_two_predictions():
    """Check that the standard deviation is based on enough values."""
    y_true = np.array([80.0])
    y_pred = np.array([81.0])

    with pytest.raises(
        ValueError,
        match='At least two predictions',
    ):
        calculate_metrics(y_true, y_pred)


def test_calculate_metrics_rejects_non_finite_values():
    """Check that NaN and infinite values are rejected."""
    y_true = np.array([80.0, np.nan])
    y_pred = np.array([81.0, 82.0])

    with pytest.raises(
        ValueError,
        match='finite numbers',
    ):
        calculate_metrics(y_true, y_pred)


def test_plot_results_creates_expected_files(tmp_path):
    """Check that evaluation plots are written to the output directory."""
    csv_path = tmp_path / 'predictions.csv'
    output_dir = tmp_path / 'plots'

    results = pd.DataFrame(
        {
            'pt_true_mev': [80.0, 85.0, 90.0],
            'pt_pred_mev': [81.0, 84.0, 91.0],
        }
    )
    results.to_csv(csv_path, index=False)

    plot_results(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
    )

    scatter_path = output_dir / 'reconstruction_scatter.png'
    residuals_path = output_dir / 'error_residuals.png'

    assert scatter_path.exists()
    assert residuals_path.exists()

    assert scatter_path.stat().st_size > 0
    assert residuals_path.stat().st_size > 0


def test_plot_results_rejects_missing_prediction_columns(tmp_path):
    '''The prediction CSV must contain both required momentum columns.'''

    csv_path = tmp_path / 'predictions.csv'
    output_dir = tmp_path / 'plots'

    pd.DataFrame({
        'pt_true_mev': [80.0, 85.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match='pt_true_mev and pt_pred_mev',
    ):
        plot_results(
            csv_path=str(csv_path),
            output_dir=str(output_dir),
        )


def test_plot_results_reports_missing_input_file(tmp_path):
    """Check that a missing prediction CSV produces a clear error."""
    missing_csv = tmp_path / 'missing_predictions.csv'
    output_dir = tmp_path / 'plots'

    with pytest.raises(
        FileNotFoundError,
        match='Run predict.py first',
    ):
        plot_results(
            csv_path=str(missing_csv),
            output_dir=str(output_dir),
        )
