import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TrackPreprocessor:
    '''
    Handles the ingestion, splitting, and safe standardization of tracking data.
    This class ensures zero data leakage by computing scaling statistics strictly on the training subset. 
    It explicitly isolates sentinel padding values during statistical calculations so the dead-zones do not distort the true feature distributions.
    '''
    def __init__(self, filepath: str, pad_val: float = -9999.0, random_state: int = 13):
        self.filepath = filepath
        self.pad_val = pad_val
        self.random_state = random_state
        
        #Internal state to hold training statistics for safe downstream scaling
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def load_and_split(self) -> dict:
        '''Loads the raw CSV and partitions it into Train, Validation, and Test sets.'''
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError as e:
            msg = f'Data file not found at {self.filepath}. Please run simulation.py first.'
            raise FileNotFoundError(msg) from None

        #Isolate the target (pt_true) from the spatial hit coordinates
        y = df['pt_true'].values.reshape(-1, 1)
        X = df.drop(columns = ['pt_true']).values

        #Isolate the Training set
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.25, random_state = self.random_state)
        #Divide the remainder into Validation and Test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.50, random_state = self.random_state)

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        '''Calculates the mean and standard deviation of the features and targets strictly from the training data.'''
        #Create a boolean mask to completely ignore padded values during stat calculation
        valid_mask = (X_train != self.pad_val)
        
        #Calculate feature-wise statistics using only valid hits.
        self.x_mean = np.zeros(X_train.shape[1])
        self.x_std = np.ones(X_train.shape[1])
        
        for col_idx in range(X_train.shape[1]):
            valid_col_data = X_train[ : , col_idx][valid_mask[ : , col_idx]]
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
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError('The preprocessor must be fitted before calling transform.')

        #Initialize scaled arrays
        X_scaled = np.copy(X)
        y_scaled = (y - self.y_mean) / self.y_std
        
        #Apply scaling only to the valid indices, utilizing NumPy broadcasting
        valid_mask = (X != self.pad_val)
        
        for col_idx in range(X.shape[1]):
            col_mask = valid_mask[ : , col_idx]
            X_scaled[col_mask, col_idx] = (X[col_mask, col_idx] - self.x_mean[col_idx]) / self.x_std[col_idx]

        #Use an assert here strictly to explicitly document the dimensional invariant
        assert X_scaled.shape[0] == y_scaled.shape[0], 'Feature and target row counts diverged during transform.'
        
        return X_scaled, y_scaled

    def save_tensors(self, data_dict: dict, outdir: str):
        '''Saves the processed arrays to disk.'''
        os.makedirs(outdir, exist_ok = True)
        
        for name, array in data_dict.items():
            outpath = os.path.join(outdir, f'{name}.npy')
            np.save(outpath, array)
            
        #Save the target statistics so predict.py can un-scale the final outputs back to MeV/c
        stats_path = os.path.join(outdir, 'y_stats.npz')
        np.savez(stats_path, y_mean = self.y_mean, y_std = self.y_std)

    def run_pipeline(self, outdir: str):
        '''Executes the entire preprocessing workflow.'''
        splits = self.load_and_split()
        
        self.fit(splits['X_train'], splits['y_train'])
        
        splits['X_train'], splits['y_train'] = self.transform(splits['X_train'], splits['y_train'])
        splits['X_val'], splits['y_val'] = self.transform(splits['X_val'], splits['y_val'])
        splits['X_test'], splits['y_test'] = self.transform(splits['X_test'], splits['y_test'])
        
        self.save_tensors(splits, outdir)
        print(f'Preprocessing complete. Processed tensors and scaling stats saved to {outdir}')


def main(): #pragma: no cover
    '''Command line interface for the preprocessing pipeline.'''
    parser = argparse.ArgumentParser(description = 'Preprocess track data for neural network training.')
    parser.add_argument('--input', type = str, default = 'data_files/simulated_data/simulated_tracks.csv', help = 'Path to the raw CSV file.')
    parser.add_argument('--outdir', type = str, default = 'data_files/processed_data', help = 'Directory to save the processed arrays.')
    parser.add_argument('--seed', type = int, default = 13, help = 'Random seed for data splitting.')

    args = parser.parse_args()
    
    processor = TrackPreprocessor(filepath = args.input, random_state = args.seed)
    
    try:
        processor.run_pipeline(args.outdir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Preprocessing failed: {exc}') from exc

if __name__ == '__main__': #pragma: no cover
    main()

#Property of Wafa Mamani. May 2026.