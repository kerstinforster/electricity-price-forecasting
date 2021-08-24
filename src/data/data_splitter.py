""" This class implements a data splitter that creates training samples,
which are ready to be used for our models"""

import pandas as pd
import numpy as np
import tensorflow as tf


class DataSplitter:
    """
    This class splits the preprocessed dataframe into training and testing
    samples. First, the dataframe
    """
    def __init__(self, params: dict):
        """
        Initialize the data splitter with the required data
        :param params: Parameter dictionary
        """
        self.window_size = params.get('window_size', 7*24)
        self.batch_size = params.get('batch_size', 64)
        self.gap = params.get('gap', 0)
        self.shuffle = params.get('shuffle', False)
        # Shuffle True seems to break stuff

    def split(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Splitting the dataset with a sliding window evaluation method
        As a queue the training set and validation value are moving on
        one position after each iteration
        Between the training set and the validation set is a gap for skipped
        values
        (+1h: gap=0; +24h: gap=23; +168h: gap = 167)

        :param data: Dataframe of dimensions (N_Timesteps, N_Features)
        :param gap: skipped values between training and validation set
        :return dataset:
        """
        data = data.drop(['Time'], axis=1)
        y_data = np.expand_dims(data.SPOTPrice.to_numpy(), 0)
        y_data = y_data[:, self.window_size+self.gap:].reshape((-1,))

        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data, y_data, sequence_length=7 * 24,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        return dataset


def train_test_split(data: pd.DataFrame, test_size: float = 0.2):
    rows = data.shape[0]
    split_index = int((1-test_size) * rows)
    return data[:split_index], data[split_index:]
