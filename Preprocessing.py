import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TrackPreprocessor:
    def __init__(self, filepath, pad_val=-9999.0, random_state=13):
        self.filepath = filepath
        self.pad_val = pad_val
        self.random_state = random_state
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def load_and_split(self):
        df = pd.read_csv(self.filepath)
        y = df['pt_true'].values.reshape(-1, 1)
        X = df.drop(columns=['pt_true']).values

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }

    def fit(self, X_train, y_train):
        self.x_mean = np.zeros(X_train.shape[1])
        self.x_std = np.ones(X_train.shape[1])
        valid_mask = X_train != self.pad_val

        for col_idx in range(X_train.shape[1]):
            vals = X_train[:, col_idx][valid_mask[:, col_idx]]
            if len(vals) > 0:
                self.x_mean[col_idx] = vals.mean()
                std = vals.std()
                self.x_std[col_idx] = std if std > 0 else 1.0

        self.y_mean = y_train.mean()
        self.y_std = y_train.std()

    def transform(self, X, y):
        X_scaled = X.copy()
        valid_mask = X != self.pad_val
        for col_idx in range(X.shape[1]):
            mask = valid_mask[:, col_idx]
            X_scaled[mask, col_idx] = (X[mask, col_idx] - self.x_mean[col_idx]) / self.x_std[col_idx]
        y_scaled = (y - self.y_mean) / self.y_std
        return X_scaled, y_scaled

    def save_tensors(self, data_dict, outdir):
        os.makedirs(outdir, exist_ok=True)
        for name, array in data_dict.items():
            np.save(os.path.join(outdir, f'{name}.npy'), array)
        np.savez(os.path.join(outdir, 'y_stats.npz'), y_mean=self.y_mean, y_std=self.y_std)

    def run_pipeline(self, outdir):
        splits = self.load_and_split()
        self.fit(splits['X_train'], splits['y_train'])
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            y_key = f'y_{split}'
            splits[X_key], splits[y_key] = self.transform(splits[X_key], splits[y_key])
        self.save_tensors(splits, outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data_files/simulated_data/simulated_tracks.csv')
    parser.add_argument('--outdir', default='data_files/processed_data')
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    processor = TrackPreprocessor(args.input, random_state=args.seed)
    processor.run_pipeline(args.outdir)
    print('saved processed data to', args.outdir)


if __name__ == '__main__':
    main()
