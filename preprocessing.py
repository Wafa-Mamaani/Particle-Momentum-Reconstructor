import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TrackPreprocessor:
    '''
    Loads tracking data, creates train, validation, and test splits, and
    standardizes the data using statistics calculated from the training set.

    The padding value is ignored when feature means and standard deviations
    are calculated.
    '''
    def __init__(self, filepath: str, pad_val: float = -9999.0, random_state: int = 13):
        self.filepath = filepath
        self.pad_val = pad_val
        self.random_state = random_state
        
        # Statistics calculated from the training set
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def load_and_split(self) -> dict:
        '''Loads the raw CSV and partitions it into Train, Validation, and Test sets.'''
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            msg = f'Data file not found at {self.filepath}. Please run simulation.py first.'
            raise FileNotFoundError(msg) from None

        if 'pt_true' not in df.columns:
            raise ValueError(
                "Input CSV is missing the required column 'pt_true'."
            )

        feature_df = df.drop(columns=['pt_true'])

        try:
            feature_df = feature_df.apply(
                pd.to_numeric,
                errors='raise',
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                'Feature columns must contain only numeric values.'
            ) from exc
        feature_array = feature_df.to_numpy(dtype=float)

        if not np.all(np.isfinite(feature_array)):
            raise ValueError(
                'Feature columns must contain only finite values.'
            )
        try:
            target_series = pd.to_numeric(
                df['pt_true'],
                errors='raise',
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
               "Target column 'pt_true' must contain only numeric values."
            ) from exc
        
        if not np.all(np.isfinite(target_series.to_numpy(dtype=float))):
            raise ValueError(
                "Target column 'pt_true' must contain only finite values."
            )

        if len(df) < 5:
            raise ValueError(
                'Input dataset must contain at least 5 rows to create '
                'train, validation, and test splits.'
            )
        
        # Isolate the target (pt_true) from the spatial hit coordinates
        y = target_series.to_numpy(dtype=float).reshape(-1, 1)
        X = feature_array

        # Keep 75% for training and split the remaining 25% equally
        # between validation and test data.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=self.random_state,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=self.random_state,
        )

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        '''Calculates the mean and standard deviation of the features and targets strictly from the training data.'''
        # Exclude padded hits from the feature statistics
        valid_mask = X_train != self.pad_val
        
        # Calculate feature-wise statistics using only valid hits.
        self.x_mean = np.zeros(X_train.shape[1])
        self.x_std = np.ones(X_train.shape[1])
        
        for col_idx in range(X_train.shape[1]):
            valid_col_data = X_train[:, col_idx][valid_mask[:, col_idx]]
            if len(valid_col_data) > 0:
                self.x_mean[col_idx] = np.mean(valid_col_data)
                std_val = np.std(valid_col_data)
                self.x_std[col_idx] = std_val if std_val > 0 else 1.0

        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train)

        if self.y_std == 0:
            raise ValueError(
                'Target standard deviation is zero; target scaling is undefined.'
            )

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple:
        '''
        Applies the training statistics to normalize the input arrays.
        Padded values are bypassed and left exactly as they are.
        '''
        if (
            self.x_mean is None
            or self.x_std is None
            or self.y_mean is None
            or self.y_std is None
        ):
            raise RuntimeError(
                'The preprocessor must be fitted before calling transform.'
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Feature and target arrays must contain the same number of rows.'
            )
        # Initialize scaled arrays
        X_scaled = np.copy(X)
        y_scaled = (y - self.y_mean) / self.y_std
        
        # Scale only the non-padded values
        valid_mask = X != self.pad_val
        
        for col_idx in range(X.shape[1]):
            col_mask = valid_mask[:, col_idx]
            X_scaled[col_mask, col_idx] = (
                X[col_mask, col_idx] - self.x_mean[col_idx]
            ) / self.x_std[col_idx]

        return X_scaled, y_scaled

    def save_tensors(self, data_dict: dict, outdir: str):
        '''Saves the processed arrays to disk.'''
        os.makedirs(outdir, exist_ok=True)
        
        for name, array in data_dict.items():
            outpath = os.path.join(outdir, f'{name}.npy')
            np.save(outpath, array)
            
        # Save the target statistics so predict.py can un-scale
        # the final outputs back to MeV/c
        stats_path = os.path.join(outdir, 'y_stats.npz')
        np.savez(
            stats_path,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

    def run_pipeline(self, outdir: str):
        '''Executes the entire preprocessing workflow.'''
        splits = self.load_and_split()
        
        self.fit(splits['X_train'], splits['y_train'])
        
        splits['X_train'], splits['y_train'] = self.transform(
            splits['X_train'],
            splits['y_train'],
        )
        splits['X_val'], splits['y_val'] = self.transform(
            splits['X_val'],
            splits['y_val'],
        )
        splits['X_test'], splits['y_test'] = self.transform(
            splits['X_test'],
            splits['y_test'],
        )
        
        self.save_tensors(splits, outdir)
        print(f'Preprocessing complete. Results saved to {outdir}')


def main():  # pragma: no cover
    '''Command line interface for the preprocessing pipeline.'''
    parser = argparse.ArgumentParser(
        description='Preprocess track data for neural network training.'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data_files/simulated_data/simulated_tracks.csv',
        help='Path to the raw CSV file.',
    )

    parser.add_argument(
        '--outdir',
        type=str,
        default='data_files/processed_data',
        help='Directory to save the processed arrays.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=13,
        help='Random seed for data splitting.',
    )

    args = parser.parse_args()
    
    processor = TrackPreprocessor(
        filepath=args.input,
        random_state=args.seed,
    )
    
    try:
        processor.run_pipeline(args.outdir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Preprocessing failed: {exc}') from exc

if __name__ == '__main__':  # pragma: no cover
    main()

# Property of Wafa Mamaani. May 2026.
