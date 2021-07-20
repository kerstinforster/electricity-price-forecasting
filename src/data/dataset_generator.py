"""This is the dataset generator combining all the data getters"""

from typing import List
import pandas as pd

from src.data.montel_data_getter import MontelDataGetter
from src.data.weather_data_getter import WeatherDataGetter


class DatasetGenerator:
    """
    This class combines the data getters to generate a dataset which
    can then be used for the time series prediction.
    """

    def __init__(self, datasets: List[str] = None):
        """
        Constructor fo the class that
        :param datasets: List of all the data types to be used
        """
        self.datasets = datasets
        if self.datasets == ['all'] or self.datasets is None:
            self.datasets = ['montel', 'weather']
        self.data_getters = \
            [DataGetterFactory.get(dataset) for dataset in self.datasets]

    def get_dataset(self, start_date: str, end_date: str, end_time: str):
        """
        Crate one large dataset by combining all the getters
        :param start_date: Start date for the dataset to be generated
        :param end_date: End date for the dataset or 'latest'
        :param end_time: End time in format 'T00' until 'T23'
        :return: A pandas dataframe containing all the data
        """
        data = []
        for getter in self.data_getters:
            # Dataset are all structured the same way:
            # First column is timestamp, the other columns are values
            # Also, the timestamps in all datasets should be the same!
            data.append(getter.get_data(start_date, end_date, end_time))
        dataset = data[0]
        if end_date == 'latest':
            data = self.adjust_dataset_lengths(data)

        for i in range(1, len(data)):
            assert len(data[i].index) == len(data[0].index)
            dataset = pd.merge_asof(dataset, data[i], on='Time')
        # Check that no value in the dataframe is NaN
        assert not dataset.isnull().values.any()
        return dataset

    def adjust_dataset_lengths(self, data: List[pd.DataFrame]) \
            -> List[pd.DataFrame]:
        """
        All datasets need the same length to be combined into one larger dataset
        If we select 'latest' as end date, it is possible to have different
        length datasets from the data getters. Therefore, this function
        selects the shortest dataset and cuts all the others to fit that length.
        :param data:
        :return:
        """
        lengths = []
        for dataset in data:
            lengths.append(len(dataset.index))
        min_length = min(lengths)
        shortened_data = []
        for dataset in data:
            shortened_data.append(dataset.iloc[0: min_length, :])
        return shortened_data


class DataGetterFactory:
    """
    Factory for data getter class instances
    """

    @staticmethod
    def get(getter_name: str):
        """
        Get a data getter based on the name of the data
        :param getter_name: Name of the getter
        :return: Class instance of the getter
        """
        if getter_name == 'montel':
            return MontelDataGetter()
        elif getter_name == 'weather':
            return WeatherDataGetter()
        else:
            raise ValueError(f'The data getter for type "{getter_name}" '
                             f'does not exist!')
