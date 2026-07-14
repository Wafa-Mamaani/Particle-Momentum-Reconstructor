import os
import numpy as np
import pandas as pd
import torch

from model import TrackMomentumRegressor


def run_inference(data_dir="data_files/processed_data", weights_path="weights/best_model.pth", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy")), dtype=torch.float32)
    y_test_scaled = np.load(os.path.join(data_dir, "y_test.npy"))

    model = TrackMomentumRegressor(input_dim=72, pad_val=-9999.0)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    with torch.no_grad():
        predictions_scaled = model(X_test).numpy()

    results = pd.DataFrame({
        "pt_true_scaled": y_test_scaled.flatten(),
        "pt_pred_scaled": predictions_scaled.flatten()
    })

    outpath = os.path.join(output_dir, "test_predictions.csv")
    results.to_csv(outpath, index=False)
    print(f"Inference results saved to {outpath}")


if __name__ == "__main__":
    run_inference()
