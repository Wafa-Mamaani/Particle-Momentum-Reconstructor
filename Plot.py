import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(csv_path="results/test_predictions.csv", output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    y_true = df["pt_true_mev"]
    y_pred = df["pt_pred_mev"]
    residuals = y_pred - y_true
    rmse = (residuals ** 2).mean() ** 0.5

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(limits, limits, linestyle="--")

    plt.xlabel("True Transverse Momentum [MeV/c]")
    plt.ylabel("Predicted Transverse Momentum [MeV/c]")
    plt.title("Tracker Momentum Reconstruction Accuracy")
    plt.plot([], [], " ", label=f"RMSE: {rmse:.3f} MeV/c")
    plt.legend()
    plt.grid(True, alpha=0.3)

    scatter_path = os.path.join(output_dir, "reconstruction_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)

    plt.xlabel("Momentum Residual (Predicted - True) [MeV/c]")
    plt.ylabel("Track Count")
    plt.title("Momentum Reconstruction Error Distribution")
    plt.grid(True, alpha=0.3)

    hist_path = os.path.join(output_dir, "error_residuals.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plot_results()
