import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# early preprocessing draft - just get data into train/test arrays
DATA_PATH = 'data_files/simulated_data/simulated_tracks.csv'
OUT_DIR = 'data_files/processed_data'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    y = df['pt_true'].values.reshape(-1, 1)
    X = df.drop('pt_true', axis=1).values
    return X, y


def make_split(X, y):
    # TODO: maybe add validation set later
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=13
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = make_split(X, y)
    print('train:', X_train.shape, y_train.shape)
    print('test:', X_test.shape, y_test.shape)
