""" This file contains a data transformer that normalizes the dataset. """

import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataTransformer:
    """
    Data Transformer class that scales the dataset
    """

    def __init__(self):
        """
        Constructor
        """
        self.scaler = preprocessing.RobustScaler()
        self.spot_scaler = preprocessing.RobustScaler()
        self.columns = None

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function transforms the dataset by normalizing all the columns.
        In the future, additional transformations might be possible.
        :param data: Dataframe that contains the dataset
        :return: Transformed dataframe
        """
        values = data.copy().drop('Time', axis='columns')
        scaled_values = pd.DataFrame(self.scaler.transform(values.values),
                                     columns=values.columns, index=values.index)
        return data[['Time']].join(scaled_values)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the dataset using a RobustScaler using the inter-quartile range
        :param data: the dataset to scale
        :return: Scaled dataset
        """
        self.columns = data.columns
        values = data.copy().drop('Time', axis='columns')
        scaled_values = pd.DataFrame(self.scaler.fit_transform(values.values),
                                     columns=values.columns, index=values.index)
        spot_values = self.spot_scaler.fit_transform(
            np.expand_dims(values.SPOTPrice.values, 1))
        scaled_values['SPOTPrice'] = spot_values
        return data[['Time']].join(scaled_values)

    def reverse_transform(self, data: pd.DataFrame):
        """
        After computing a prediction on transformed data, use this function
        to revert the transform for the predicted data.
        :param data: The predicted data to be transformed to the original
            value range.
        :return: Transformed predicted data
        """
        values = data.copy().drop('Time', axis='columns')
        reverse_scaled_values = pd.DataFrame(
            self.scaler.inverse_transform(values.values),
            columns=values.columns, index=values.index)
        return data[['Time']].join(reverse_scaled_values)

    def reverse_transform_spot(self, data: np.ndarray):
        """
        Reverse transform only for the spot data
        :param data: Array of Spot Price values shape (n, 1)
        :return: Array of reverse transformed SPOT Price values
        """
        if not len(data.shape) == 2:
            data = np.reshape(data, (-1, 1))
        return self.spot_scaler.inverse_transform(data)

