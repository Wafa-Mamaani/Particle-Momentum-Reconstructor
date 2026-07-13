import os
import numpy as np
import pandas as pd
import torch

from model import TrackMomentumRegressor


def run_inference(data_dir="data_files/processed_data", weights_path="weights/best_model.pth", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy")), dtype=torch.float32)

    model = TrackMomentumRegressor()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    with torch.no_grad():
        predictions = model(X_test).numpy()

    results = pd.DataFrame({
        "pt_pred": predictions.flatten()
    })

    outpath = os.path.join(output_dir, "test_predictions.csv")
    results.to_csv(outpath, index=False)
    print("saved predictions to", outpath)


if __name__ == "__main__":
    run_inference()
