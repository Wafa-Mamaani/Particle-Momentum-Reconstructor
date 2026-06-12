import argparse
import os

import numpy as np
import pandas as pd

MISSING_VALUE = -9999.0
DEFAULT_LAYERS = np.linspace(0.385, 0.695, 36).tolist()


def generate_toy_tracks(
    num_samples: int,
    b_field: float = 1.0,
    layers: list = DEFAULT_LAYERS,
    efficiency: float = 0.95,
    resolution: float = 0.001,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate toy charged-particle tracks in a simple detector geometry."""
    if num_samples <= 0:
        raise ValueError("num_samples should be positive")

    rng = np.random.default_rng(random_seed)
    r_layers = np.array(layers)
    num_layers = len(r_layers)

    pt_true = rng.uniform(65.0, 105.0, num_samples)
    alpha = rng.uniform(0, 2 * np.pi, num_samples)
    charge = rng.choice([-1, 1], num_samples)

    radius = (pt_true / 1000) / (0.3 * b_field)
    phi_center = alpha + charge * (np.pi / 2)

    x_hits = np.full((num_samples, num_layers), MISSING_VALUE)
    y_hits = np.full((num_samples, num_layers), MISSING_VALUE)

    for layer_index, layer_radius in enumerate(r_layers):
        valid_geometry_mask = layer_radius <= (2 * radius)
        beta = np.arccos(layer_radius / (2 * radius[valid_geometry_mask]))
        theta_hit = phi_center[valid_geometry_mask] - charge[valid_geometry_mask] * beta

        x_true = layer_radius * np.cos(theta_hit)
        y_true = layer_radius * np.sin(theta_hit)

        hit_recorded_mask = rng.random(np.sum(valid_geometry_mask)) < efficiency
        global_indices = np.where(valid_geometry_mask)[0][hit_recorded_mask]

        x_hits[global_indices, layer_index] = x_true[hit_recorded_mask] + rng.normal(
            0, resolution, np.sum(hit_recorded_mask)
        )
        y_hits[global_indices, layer_index] = y_true[hit_recorded_mask] + rng.normal(
            0, resolution, np.sum(hit_recorded_mask)
        )

    columns = ["pt_true"]
    arrays = [pt_true.reshape(-1, 1)]
    for i in range(num_layers):
        columns.extend([f"hit_{i}_x", f"hit_{i}_y"])
        arrays.append(x_hits[:, i].reshape(-1, 1))
        arrays.append(y_hits[:, i].reshape(-1, 1))

    return pd.DataFrame(np.hstack(arrays), columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Generate toy tracking data.")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--outdir", type=str, default="data_files/simulated_data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, "simulated_tracks.csv")

    print(f"Generating {args.samples} tracks")
    df = generate_toy_tracks(num_samples=args.samples, random_seed=args.seed)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
