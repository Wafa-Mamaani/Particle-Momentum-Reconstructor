import os
import argparse
import numpy as np
import pandas as pd
import torch

from model import TrackMomentumRegressor

def run_inference(data_dir: str, weights_path: str, output_dir: str):
    """
    Loads the trained model weights and evaluates the held-out test set.
    Reverses the statistical scaling to output physical momentum values (MeV/c).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy")), dtype=torch.float32).to(device)
        y_test_scaled = np.load(os.path.join(data_dir, "y_test.npy"))

        stats = np.load(os.path.join(data_dir, "y_stats.npz"))
        y_mean = stats["y_mean"]
        y_std = stats["y_std"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing processed data or stats. "
            f"Run preprocessing first. Details: {exc}"
        ) from exc

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Run train.py first."
        )

    model = TrackMomentumRegressor(
        input_dim=72,
        pad_val=-9999.0,
    ).to(device)

    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        predictions_scaled = model(X_test).cpu().numpy()

    predictions_mev = (predictions_scaled * y_std) + y_mean
    y_test_mev = (y_test_scaled * y_std) + y_mean

    results_df = pd.DataFrame({
        "pt_true_mev": y_test_mev.flatten(),
        "pt_pred_mev": predictions_mev.flatten()
    })

    outpath = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(outpath, index=False)
    print(f"Inference complete. Physical predictions saved to {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test set.")
    parser.add_argument("--data", type=str, default="data_files/processed_data", help="Directory containing X_test.npy.")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Path to the .pth weights.")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save the prediction CSV.")

    args = parser.parse_args()
    run_inference(args.data, args.weights, args.outdir)


if __name__ == "__main__":
    main()


# Property of Wafa Mamani. May 2026.
