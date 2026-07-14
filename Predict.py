import os
import numpy as np
import pandas as pd
import torch

from model import TrackMomentumRegressor


def run_inference(data_dir="data_files/processed_data", weights_path="weights/best_model.pth", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy")), dtype=torch.float32).to(device)
        y_test_scaled = np.load(os.path.join(data_dir, "y_test.npy"))

        stats = np.load(os.path.join(data_dir, "y_stats.npz"))
        y_mean = stats["y_mean"]
        y_std = stats["y_std"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing processed data or scaling statistics. Run preprocessing first. Details: {exc}")

    model = TrackMomentumRegressor(input_dim=72, pad_val=-9999.0).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        predictions_scaled = model(X_test).cpu().numpy()

    predictions_mev = (predictions_scaled * y_std) + y_mean
    y_test_mev = (y_test_scaled * y_std) + y_mean

    results = pd.DataFrame({
        "pt_true_mev": y_test_mev.flatten(),
        "pt_pred_mev": predictions_mev.flatten()
    })

    outpath = os.path.join(output_dir, "test_predictions.csv")
    results.to_csv(outpath, index=False)
    print(f"Inference complete. Predictions saved to {outpath}")


if __name__ == "__main__":
    run_inference()
