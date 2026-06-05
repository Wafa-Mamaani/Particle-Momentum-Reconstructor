import numpy as np
import pandas as pd

MISSING_VALUE = -9999.0


def generate_toy_tracks(
    num_samples: int,
    b_field: float = 1.0,
    layers=None,
    efficiency: float = 0.95,
    resolution: float = 0.001,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a rectangular toy dataset of particle hits."""
    if layers is None:
        layers = np.linspace(0.385, 0.695, 36)

    # TODO: change this to default_rng later
    np.random.seed(random_seed)

    r_layers = np.array(layers)
    num_layers = len(r_layers)

    pt_true = np.random.uniform(65.0, 105.0, num_samples)
    alpha = np.random.uniform(0, 2 * np.pi, num_samples)
    charge = np.random.choice([-1, 1], num_samples)

    radius = (pt_true / 1000) / (0.3 * b_field)
    phi_center = alpha + charge * (np.pi / 2)

    x_hits = np.full((num_samples, num_layers), MISSING_VALUE)
    y_hits = np.full((num_samples, num_layers), MISSING_VALUE)

    for i, r_i in enumerate(r_layers):
        valid = r_i <= (2 * radius)
        valid_ids = np.where(valid)[0]

        beta = np.arccos(r_i / (2 * radius[valid]))
        theta_hit = phi_center[valid] - charge[valid] * beta

        x_true = r_i * np.cos(theta_hit)
        y_true = r_i * np.sin(theta_hit)

        # add simple detector inefficiency and measurement noise
        keep = np.random.random(len(valid_ids)) < efficiency
        kept_ids = valid_ids[keep]

        x_hits[kept_ids, i] = x_true[keep] + np.random.normal(0, resolution, np.sum(keep))
        y_hits[kept_ids, i] = y_true[keep] + np.random.normal(0, resolution, np.sum(keep))

    df = pd.DataFrame({"pt_true": pt_true})
    for i in range(num_layers):
        # not the most efficient way, but readable enough for now
        df[f"hit_{i}_x"] = x_hits[:, i]
        df[f"hit_{i}_y"] = y_hits[:, i]

    return df


if __name__ == "__main__":
    df = generate_toy_tracks(10000, random_seed=13)
    df.to_csv("simulated_tracks.csv", index=False)
