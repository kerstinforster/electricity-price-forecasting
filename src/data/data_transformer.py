""" This file contains a data transformer that normalizes the dataset. """

import pandas as pd
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

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function transforms the dataset by normalizing all the columns.
        In the future, additional transformations might be possible.
        :param data: Dataframe that contains the dataset
        :return: Transformed dataframe
        """
        return self._scale(data)

    def reverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        After computing a prediction on transformed data, use this function
        to revert the transform for the predicted data.
        :param data: The predicted data to be transformed to the original
            value range.
        :return: Transformed predicted data
        """
        return self._revert_scale(data)

    def _scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the dataset using a RobustScaler using the inter-quartile range
        :param data: the dataset to scale
        :return: Scaled dataset
        """
        values = data.copy().drop('Time', axis='columns')
        scaled_values = pd.DataFrame(self.scaler.fit_transform(values.values),
                                     columns=values.columns, index=values.index)
        return data[['Time']].join(scaled_values)

    def _revert_scale(self, data):
        """
        Revert the scaling using the saved parameters
        :param data: The data re-transformed to its original scale
        :return:
        """
        values = data.copy().drop('Time', axis='columns')
        reverse_scaled_values = pd.DataFrame(
            self.scaler.inverse_transform(values.values),
            columns=values.columns, index=values.index)
        return data[['Time']].join(reverse_scaled_values)
