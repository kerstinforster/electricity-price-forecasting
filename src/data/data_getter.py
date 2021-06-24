""" This file contains the base class to the data getters."""

from typing import Any
from abc import ABC, abstractmethod
from pathlib import Path
import os
import numpy as np
from datetime import datetime
import pandas as pd


class BaseDataGetter(ABC):
    """
    The abstract interface for all data getters
    """
    def __init__(self, name: str):
        """
        Constructor for the data getter setting the name field
        :param name: Name of the data getter
        """
        self.name = name
        self._root_dir = self.get_project_root()
        self.data_dir = os.path.join(self._root_dir, 'data', name)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.start_date = '2012-01-01'
        self.end_date = '2021-06-01'
        self.end_time = 'T23'

    def get_data(self, start_date: str = '2012-01-01',
                 end_date: str = '2021-06-01', end_time: str = 'T23',
                 overwrite: bool = False) -> pd.DataFrame:
        """
        This is the entry point for any program. This is the only function
        that should be called from the outside.
        This function is responsible for getting all the required data between
        the start and end date. Checks if the data is already downloaded and
        loads all the data in a dataframe
        :param start_date: Start date to get the data from
        :param end_date: End date to get the data for (included in data) or
        'latest' (get as new data as possible)
        :param end_time: End time in format 'T00' until 'T23'
        :param overwrite: Bool flag to force downloading the data
        :return: pd.DataFrame containing the data and timestamps
        """
        self.start_date = start_date
        self.end_date = end_date
        self.end_time = end_time
        if overwrite or end_date == 'latest' or\
                not os.path.exists(os.path.join(self.data_dir, 'data.csv')):
            self.download_data()
        return self.load_data()

    def download_data(self) -> None:
        """
        This function needs to download the data from a given start date until
        a given end date. It is required to use hourly precision datasets only!
        make sure to handle summer and winter time -> every day should have
        exactly 24 hours - Drop or add values if necessary.

        The data is then stored in a csv file in the form:
        | timestamp (day + hour) | value1 | valueN (optional) |
        """
        print(f'Downloading {self.name} data.')
        data = self._get_raw_data()
        processed_data = self._process_raw_data(data)
        df = pd.DataFrame(data=processed_data)
        df['Time'] = pd.to_datetime(df['Time'])
        df.to_csv(os.path.join(self.data_dir, 'data.csv'), index=False)

    def load_data(self) -> pd.DataFrame:
        """
        Load the required data from the csv file in the data dir.
        Then, filter only the required dates between start and end date
        The data fields need to look like:
        | Time (day + hour) | value1 | valueN (optional) |
        :return: Dataframe containing the data between the start and end dates
        """
        all_data = pd.read_csv(os.path.join(self.data_dir, 'data.csv'))
        all_data['Time'] = pd.to_datetime(all_data['Time'])
        # Get slice indices
        start = pd.to_datetime(self.start_date + 'T00')
        start_index = all_data.index[all_data['Time'] == start]
        end_index = np.array([None]) if self.end_date == 'latest' else \
            all_data.index[all_data['Time'] ==
                           pd.to_datetime(self.end_date + self.end_time)] + 1
        if not start_index.size or not end_index.size:
            # Downloaded data does not contain all the required data
            self.download_data()
            return self.load_data()
        data = all_data.iloc[start_index[0]:end_index[0], :]
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def get_project_root() -> Path:
        """
        This function returns the project's root directory
        :return: The root path of the project
        """
        return Path(__file__).parent.parent.parent

    @abstractmethod
    def _get_raw_data(self) -> Any:
        """
        This function must be implemented in a derived class to download the
        raw data from the data source.
        After downloading this data, it is then processed and finally stored in
        other functions within this class
        :param start_date: The first date for which to get data, form yyyy-mm-dd
        :param end_date: The last date for which to get data, form yyyy-mm-dd
        :return: The raw data in any format
        """
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
    def _process_raw_data(self, data: Any) -> Any:
        """
        This function must be implemented in a derived class to process the
        raw downloaded data from the data source.
        After processing, the data must be in a form to create a dataframe from
        it like so: pd.DataFrame(data=processed_data).
        One way of formatting the processed data is to create a list of dicts,
        where each key in the dicts is a column in the dataframe.
        Suggested: list [ dict {'Date': 'yyyy-mm-dd', 'Hour': num, 'Value': v}]
        :param data: The raw data from the data source
        :return: The processed data
        """
        raise NotImplementedError('This is an abstract class method.')

    def _get_num_days(self):
        """
        Returns the number of days between two dates.
        Both the start end the end date are included in the number of days!
        :param start_date: Start date in form yyyy-mm-dd
        :param end_date: End date in form yyyy-mm-dd
        :return:
        """
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        if self.end_date == 'latest':
            now = datetime.now()
            end = datetime.strptime(now.strftime('%Y-%m-%d'), '%Y-%m-%d')
        else:
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
        delta = end - start
        return delta.days + 1
