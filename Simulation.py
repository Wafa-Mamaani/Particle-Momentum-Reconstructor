import numpy as np
import pandas as pd

# Still a toy simulation. Now trying to include magnetic bending.
# Some constants are still hardcoded while the logic settles.

B_FIELD = 1.0
MISSING_VALUE = -9999.0


def generate_toy_tracks(num_samples, random_seed=42):
    np.random.seed(random_seed)

    layers = np.linspace(0.38, 0.70, 12)
    pt_true = np.random.uniform(65.0, 105.0, num_samples)
    alpha = np.random.uniform(0, 2 * np.pi, num_samples)
    charge = np.random.choice([-1, 1], num_samples)

    # pT is in MeV here, convert to GeV for the curvature expression
    radius = (pt_true / 1000.0) / (0.3 * B_FIELD)

    rows = []
    for j in range(num_samples):
        row = {"pt_true": pt_true[j]}
        phi_center = alpha[j] + charge[j] * np.pi / 2

        for i, r in enumerate(layers):
            # curl before this detector layer
            if r > 2 * radius[j]:
                row[f"hit_{i}_x"] = MISSING_VALUE
                row[f"hit_{i}_y"] = MISSING_VALUE
                continue

            beta = np.arccos(r / (2 * radius[j]))
            theta_hit = phi_center - charge[j] * beta

            row[f"hit_{i}_x"] = r * np.cos(theta_hit)
            row[f"hit_{i}_y"] = r * np.sin(theta_hit)

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = generate_toy_tracks(1000)
    data.to_csv("simulated_tracks.csv", index=False)
    print("done")
