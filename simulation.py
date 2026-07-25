import os
import argparse
import numpy as np
import pandas as pd

def generate_toy_tracks(
    num_samples: int,
    b_field: float = 1.0,
    #36 simulated layers between the 0.38m dead-zone boundary and 0.7m outer boundary
    layers: list = np.linspace(0.385, 0.695, 36).tolist(),
    efficiency: float = 0.95,
    resolution: float = 0.001,
    random_seed: int = 42) -> pd.DataFrame:
    
    '''
    Simulates charged particle tracks in a uniform axial magnetic field.
    The particles originate from the origin (0,0) and leave spatial hits as they intersect concentric detector layers. The number of hits per track varies due  
    to geometric curling (low pT tracks don't reach outer layers) and stochastic detector inefficiencies.
    Parameters
    ----------
    num_samples : int
        Number of particle tracks to generate.
    b_field : float
        Magnetic field strength in Tesla.
    layers : list of float
        Radii of the concentric tracking layers in meters.
    efficiency : float
        Probability [0, 1] that a valid intersection produces a recorded hit.
    resolution : float
        Standard deviation of the Gaussian spatial noise in meters.
    random_seed : int
        Seed for the random number generator to ensure reproducibility.
    Returns
    -------
    pd.DataFrame
        A rectangular dataframe containing the true pT and the (x, y) coordinates for each layer. Missing or dropped hits are padded with -9999.0.
    '''
    
    if num_samples <= 0:
        raise ValueError('The number of samples must be a positive integer.')
    
    rng = np.random.default_rng(random_seed)
    r_layers = np.array(layers)
    num_layers = len(r_layers)

    #1. Sample Kinematics
    #pT range chosen to produce a mix of curling and fully reconstructed tracks
    pt_true = rng.uniform(65.0, 105.0, num_samples)
    #Emission angle uniformly distributed in [0, 2pi)
    alpha = rng.uniform(0, 2 * np.pi, num_samples)
    #Only integer charges of ±1 are considered for simplicity (positrons and electrons)
    charge = rng.choice([-1, 1], num_samples)

    #Radius of curvature from Lorentz force and circular motion
    radius = (pt_true / 1000) / (0.3 * b_field)

    #The center of the circular track lies at distance R from the origin, orthogonal to the initial momentum vector.
    #Angle of the track center relative to the origin:
    phi_center = alpha + charge * (np.pi / 2)

    #Initialize coordinate arrays filled with the sentinel padding value
    x_hits = np.full((num_samples, num_layers), -9999.0)
    y_hits = np.full((num_samples, num_layers), -9999.0)

    #2. Calculate Exact Intersections
    #For a track passing through the origin, the origin (O), the track center (C), and the hit intersection point (H) form an isosceles triangle with sides R, R, and r_layer.
    for i, r_i in enumerate(r_layers):
        #Geometric acceptance: r_i must be <= 2R. If r_i > 2R, the track curls before the layer.
        valid_geometry_mask = r_i <= (2 * radius)
        #Calculate intersection angle only for tracks that reach this layer
        cos_beta = r_i / (2 * radius[valid_geometry_mask])
        beta = np.arccos(cos_beta)
        #The angle of the intersection point from the origin is phi_center - beta. (subtracting beta aligns with the forward-time propagation of the particle).
        theta_hit = phi_center[valid_geometry_mask] - charge[valid_geometry_mask] * beta
        
        #True intersection coordinates
        x_true = r_i * np.cos(theta_hit)
        y_true = r_i * np.sin(theta_hit)
        
        #3. Apply Stochastic Inefficiency
        #Randomly drop hits to simulate dead wires / gas fluctuations
        efficiency_roll = rng.random(np.sum(valid_geometry_mask))
        hit_recorded_mask = efficiency_roll < efficiency
        
        #4. Apply Spatial Resolution (Smearing)
        x_smeared = x_true[hit_recorded_mask] + rng.normal(0, resolution, np.sum(hit_recorded_mask))
        y_smeared = y_true[hit_recorded_mask] + rng.normal(0, resolution, np.sum(hit_recorded_mask))
        
        #Map the valid, recorded, smeared hits back to the global array indices
        global_indices = np.where(valid_geometry_mask)[0][hit_recorded_mask]
        x_hits[global_indices, i] = x_smeared
        y_hits[global_indices, i] = y_smeared

    #5. Format as a Rectangular DataFrame
    columns = ['pt_true']
    data_arrays = [pt_true.reshape(-1, 1)]
    
    for i in range(num_layers):
        columns.extend([f'hit_{i}_x', f'hit_{i}_y'])
        data_arrays.append(x_hits[ : , i].reshape(-1, 1))
        data_arrays.append(y_hits[ : , i].reshape(-1, 1))
        
    final_data = np.hstack(data_arrays)
    df = pd.DataFrame(final_data, columns = columns)
    
    return df

def main(): #pragma: no cover
    '''Command line interface for generating the dataset.'''
    parser = argparse.ArgumentParser(description = 'Generate toy track data for momentum estimation.')
    parser.add_argument('--samples', type = int, default = 10000, help = 'Number of tracks to generate.')
    parser.add_argument('--seed', type = int, default = 13, help = 'Random seed for reproducibility.')
    parser.add_argument('--outdir', type = str, default = 'data_files/simulated_data', help = 'Output directory for the CSV.')

    args = parser.parse_args()
    
    print(f'Generating {args.samples} simulated tracks...')
    df = generate_toy_tracks(num_samples = args.samples, random_seed = args.seed)
    
    #Ensure output directory exists (I use relative path, instead of hardcoding some absolute path.)
    os.makedirs(args.outdir, exist_ok = True)
    outpath = os.path.join(args.outdir, 'simulated_tracks.csv')
    
    df.to_csv(outpath, index = False)
    print(f'Dataset successfully saved to {outpath}')

if __name__ == '__main__': #pragma: no cover
    main()

#Property of Wafa Mamani. May 2026.