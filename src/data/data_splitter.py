""" This class implements a data splitter that creates training samples,
which are ready to be used for our models"""

import pandas as pd
import numpy as np


class DataSplitter:
    """
    This class splits the preprocessed dataframe into training and testing
    samples. First, the dataframe
    """
    def __init__(self, window_size: int):
        self.window_size = window_size

    def split(self, data: pd.DataFrame, gap: int):
        """
        Splitting the dataset with a sliding window evaluation method
        As a queue the training set and validation value are moving on
        one position after each iteration
        Between the training set and the validation set is a gap for skipped
        values
        (+1h: gap=0; +24h: gap=23; +168h: gap = 167)

        :param data: Dataframe of dimensions (N_Timesteps, N_Features)
        :param gap: skipped values between training and validation set
        :return X_split: np.array (n_samples X n_features X window_length)
        :return: Y_split: np.array (n_samples X n_features)
        """
        data = data.drop(['Time'], axis=1)
        x_data = data.T.to_numpy()
        y_data = np.expand_dims(data.SPOTPrice.to_numpy(), 0)
        x = []
        y = []
        for i in range(x_data.shape[1] - self.window_size - gap):
            x.append(x_data[:, i:self.window_size + i])
            y.append(y_data[:, i+self.window_size+gap:i+self.window_size+gap+1])
        x_split = np.concatenate([x], axis=0)
        y_split = np.concatenate([y], axis=0)

        return x_split, y_split[:, :, 0]


def train_test_split(data: pd.DataFrame, test_size: float = 0.2):
    rows = data.shape[0]
    split_index = int((1-test_size) * rows)
    return data[:split_index], data[split_index:]
