import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = 'data_files/simulated_data/simulated_tracks.csv'
OUT_DIR = 'data_files/processed_data'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    y = df['pt_true'].values.reshape(-1, 1)
    X = df.drop(columns=['pt_true']).values
    return X, y


def make_split(X, y, seed=13):
    # first reserve a temporary holdout block
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_basic(X_train, X_val, X_test, y_train, y_val, y_test):
    # rough version: this still treats padded -9999 values as real numbers
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = y_train.mean()
    y_std = y_train.std()

    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    X, y = load_data()
    arrays = make_split(X, y)
    arrays = standardize_basic(*arrays)
    print('preprocessing draft complete')
