import os
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Calculate residuals, bias, standard deviation, and RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            'True and predicted values must have the same shape.'
        )

    if y_true.size < 2:
        raise ValueError(
            'At least two predictions are required to calculate metrics.'
        )

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError(
            'True and predicted values must contain only finite numbers.'
        )

    residuals = y_pred - y_true
    bias = float(np.mean(residuals))
    standard_deviation = float(np.std(residuals, ddof=1))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return residuals, bias, standard_deviation, rmse

def plot_results(csv_path: str, output_dir: str):
    """Generates publication-ready visualizations of the model's predictive performance strictly from the pre-calculated results CSV."""
    os.makedirs(output_dir, exist_ok = True)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f'Cannot find {csv_path}. Run predict.py first.'
        ) from exc

    y_true = df['pt_true_mev'].to_numpy(dtype=float)
    y_pred = df['pt_pred_mev'].to_numpy(dtype=float)

    residuals, res_mean, res_std, rmse = calculate_metrics(
        y_true,
        y_pred,
    )

    #Plot 1: True vs Predicted Momentum
    plt.figure(figsize = (8, 8))
    plt.scatter(y_true, y_pred, alpha = 0.5, s = 10, color = 'blue', label = 'Predicted Tracks')

    #Perfect prediction diagonal line
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(limits, limits, color = 'red', linestyle = '--', label = 'Perfect Reconstruction')

    plt.xlabel('True Transverse Momentum [MeV/c]')
    plt.ylabel('Predicted Transverse Momentum [MeV/c]')
    plt.title('Tracker Momentum Reconstruction Accuracy')
    plt.plot([], [], ' ', label = f'RMSE: {rmse:.3f} MeV/c')
    plt.legend()
    plt.grid(True, alpha = 0.3)

    scatter_path = os.path.join(output_dir, 'reconstruction_scatter.png')
    plt.savefig(scatter_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    #Plot 2: Residuals Histogram
    plt.figure(figsize = (10, 6))
    plt.hist(residuals, bins = 50, color = 'purple', alpha = 0.7, edgecolor = 'black')

    plt.xlabel('Momentum Residual (Predicted - True) [MeV/c]')
    plt.ylabel('Track Count')
    plt.title('Momentum Reconstruction Error Distribution')
    plt.axvline(0, color = 'black', linestyle = 'dashed', linewidth = 2)
    plt.plot([], [], ' ', label = f'Mean: {res_mean:.3f} MeV/c')
    plt.plot([], [], ' ', label = f'Std Dev: {res_std:.3f} MeV/c')
    plt.legend(loc='upper right')
    plt.grid(True, alpha = 0.3)

    hist_path = os.path.join(output_dir, 'error_residuals.png')
    plt.savefig(hist_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

    print(f'Plots successfully generated and saved to {output_dir}/')

def main():
    parser = argparse.ArgumentParser(
        description='Plot model evaluation results.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/test_predictions.csv',
        help='Path to predictions CSV.',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='results/plots',
        help='Directory to save the PNG images.',
    )

    args = parser.parse_args()
    plot_results(args.input, args.outdir)


if __name__ == '__main__':
    main()

#Property of Wafa Mamani. May 2026.
