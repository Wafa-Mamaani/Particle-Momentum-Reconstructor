import numpy as np
import pandas as pd

# first quick attempt to make fake detector hits
# TODO: check the physics more carefully later

def generate_tracks(n=1000):
    np.random.seed(42)

    layers = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    rows = []

    for _ in range(n):
        pt = np.random.uniform(60, 110)
        angle = np.random.uniform(0, 2 * np.pi)
        row = {"pt_true": pt}

        # very rough placeholder: straight-ish hits with some noise
        # not using magnetic field yet
        for i, r in enumerate(layers):
            x = r * np.cos(angle) + np.random.normal(0, 0.002)
            y = r * np.sin(angle) + np.random.normal(0, 0.002)
            row[f"hit_{i}_x"] = x
            row[f"hit_{i}_y"] = y

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_tracks(1000)
    df.to_csv("tracks.csv", index=False)
    print("saved tracks.csv")
