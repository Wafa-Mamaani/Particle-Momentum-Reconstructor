import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PAD_VAL = -9999.0


def load_and_split(filepath, seed=13):
    df = pd.read_csv(filepath)
    y = df['pt_true'].values.reshape(-1, 1)
    X = df.drop(columns=['pt_true']).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def fit_scalers(X_train, y_train, pad_val=PAD_VAL):
    x_mean = np.zeros(X_train.shape[1])
    x_std = np.ones(X_train.shape[1])

    # now ignore missing hit padding values when fitting feature stats
    for j in range(X_train.shape[1]):
        valid_values = X_train[:, j][X_train[:, j] != pad_val]
        if len(valid_values):
            x_mean[j] = valid_values.mean()
            std = valid_values.std()
            x_std[j] = std if std > 0 else 1.0

    y_mean = y_train.mean()
    y_std = y_train.std()
    return x_mean, x_std, y_mean, y_std


def transform(X, y, x_mean, x_std, y_mean, y_std, pad_val=PAD_VAL):
    X_scaled = X.copy()
    for j in range(X.shape[1]):
        mask = X[:, j] != pad_val
        X_scaled[mask, j] = (X[mask, j] - x_mean[j]) / x_std[j]
    y_scaled = (y - y_mean) / y_std
    return X_scaled, y_scaled


if __name__ == '__main__':
    input_file = 'data_files/simulated_data/simulated_tracks.csv'
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split(input_file)
    x_mean, x_std, y_mean, y_std = fit_scalers(X_train, y_train)

    X_train, y_train = transform(X_train, y_train, x_mean, x_std, y_mean, y_std)
    X_val, y_val = transform(X_val, y_val, x_mean, x_std, y_mean, y_std)
    X_test, y_test = transform(X_test, y_test, x_mean, x_std, y_mean, y_std)

    print(X_train.shape, X_val.shape, X_test.shape)
