""" This file contains a data differencer that differences the dataset. """

import pandas as pd

class DataDifferencer:
    """
    Data Differencer that that differences the dataset
    """

    def __init__(self):
        """
        Constructor
        """
        self.last_entry = 0
    def difference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function differences the dataset by substracting one
        entry from the next entry.
        :param data: Dataframe that contains the dataset
        :return: Differenced dataframe
        """
        return self._difference(data)

    def reverse_difference(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        After computing a prediction on transformed data, use this function
        to revert the differencing for the predicted data.
        :param data: The predicted data to be differenced.
        :return: differenced predicted data
        """
        return self._revert_difference(data)

    def _difference(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Difference the dataset using the pandas.diff function.
        Differencing the dataset leads to losing the first
        row of the original dataset.

        :param data: the dataset to difference
        :return: Differenced dataset
        """
        values = data.copy().drop('Time', axis='columns')
        self.last_entry = values.values[-1]
        differenced_values = values.diff()
        differenced_values.dropna(inplace=True)
        return data[['Time']][1:].join(
            differenced_values).reset_index(drop=True)

    def _revert_difference(self, data):
        """
        Reverse the differncing by using the stored last value
        of the original dataset.
        :param data: The predicted data that needs to be re-differenced
        :return: re-differenced data
        """
        values = data.copy().drop('Time', axis='columns')
        reverse_differenced_values = pd.DataFrame(columns=values.columns,
                                                  index=values.index)
        for index in values.index:
            reverse_differenced_values.iloc[index] = self.last_entry \
                                + values.iloc[:index+1].sum()
        return data[['Time']].join(reverse_differenced_values)

