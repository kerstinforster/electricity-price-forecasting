""" This class implements a data getter that downloads the montel electricity
stock market price data"""

import requests
import json
import os
from typing import Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.data_getter import BaseDataGetter


class MontelDataGetter(BaseDataGetter):
    """
    This class is responsible for downloading the Montel EPEX Spot data.
    This data contains electricity spot market prices in an hourly interval.
    The values are given in the unit EURO/MWh.
    Earliest date is: '2000-06-16'
    """

    def __init__(self, name: str = 'montel'):
        """
        Constructor for the Montel Data Getter
        """
        super().__init__(name)
        self.token = None
        self.get_token()
        self.now_date = datetime.now().strftime('%Y-%m-%d')

    def get_token(self, force: bool = False) -> None:
        """
        Function that tries getting the token from either the MWN server
        or an environment variable.
        :param force: Flag used to control infinite loop between check and get
        """
        file_path = os.path.join(self._root_dir, 'src', 'data',
                                 'MONTEL_TOKEN.txt')
        if not os.path.exists(file_path):
            try:
                token = json.loads(
                    requests.get(
                        'https://coop.eikon.tum.de/mbt/mbt.json',
                        timeout=10).text)[
                    'access_token']
            except Exception as e:  # pylint: disable=broad-except
                try:
                    token = os.environ['MONTEL_TOKEN']
                except KeyError as e:
                    raise BearerTokenMissingError from e
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(token)
        with open(file_path, 'r', encoding='utf-8') as file:
            self.token = file.readline()
        self._token_check(force)

    def _token_check(self, force: bool = False) -> None:
        """
        This function checks whether the token is still valid
        :raise BearerTokenMissingError: if Bearer Token invalid
        :param force: Enforce toking working this time else throw error
        """
        response = requests.get(
            'https://api.montelnews.com/spot/getmetadata',
            headers={'Authorization': f'Bearer {self.token}'})
        if response.status_code != 200:
            print(f'The MontelBearer Token seems to be invalid, '
                  f'status {response.status_code}, \nresponse:'
                  f' {response.text} \nTrying to download updated token now!\n')
            token_path = os.path.join(self._root_dir, 'src', 'data',
                                      'MONTEL_TOKEN.txt')
            os.remove(token_path)
            if not force:
                self.get_token(force=True)
            else:
                raise BearerTokenMissingError

    def _show_available_datasets(self) -> None:
        """
        This function can be used to print all available datasets provided by
        Montel.
        """
        response = requests.get(
            'https://api.montelnews.com/spot/getmetadata',
            headers={'Authorization': f'Bearer {self.token}'})
        results = json.loads(response.text)['Elements']
        for result in results:
            # All fields are: 'SpotKey', 'SpotName', 'SourceName', 'PeakType',
            # 'Country', 'DefaultCurrency', 'AvailableCurrencies'
            print(result)

    def _get_raw_data(self) -> Any:
        """
        This function downloads the datasets that shall be used for training.
        :return: json data from the API
        """
        params = {
            'spotKey': 14,  # This is the key we should use
            'fields': ['Base', 'Peak', 'Hours'],
            'fromDate': self.start_date,  # yyyy-mm-dd
            'toDate': self.now_date if self.end_date == 'latest'
            else self.end_date,  # yyyy-mm-dd
            'currency': 'eur',
            'sortType': 'ascending'
        }
        response = requests.get(
            'https://api.montelnews.com/spot/getprices',
            headers={'Authorization': f'Bearer {self.token}'},
            params=params
        )
        return response.json()

    def _process_raw_data(self, data: Any) -> Any:
        """
        Data preprocessing method that transforms the json response into more
        useful and quicker-to-access data
        :param data: The raw json data from the API
        :return: list of dicts that is suitable for pandas
        """
        processed_data = []
        data_el = data['Elements']
        end_index = -1 if self.end_date == 'latest' else None
        for element in data_el[:end_index]:
            element['Date'] = element['Date'][:10]

            # For the changes from summer to winter time we don't have 24 hours
            # For simplicity let us just always have 24 hours
            if len(element['TimeSpans']) == 23:
                element['TimeSpans'].append(element['TimeSpans'][-1])
            elif len(element['TimeSpans']) == 25:
                element['TimeSpans'] = element['TimeSpans'][:24]
            elif len(element['TimeSpans']) != 24:
                raise ValueError(f'Element {element["Date"]} has '
                                 f'{len(element["TimeSpans"])} hours.')

            for index, value in enumerate(element['TimeSpans']):
                data_point = {'Time': f'{element["Date"]}T{index:02d}',
                              'SPOTPrice': value['Value']}
                processed_data.append(data_point)

        # Handle latest end date
        if self.end_date == 'latest':
            element = data_el[-1]
            element['Date'] = element['Date'][:10]
            for index, value in enumerate(element['TimeSpans']):
                data_point = {'Time': f'{element["Date"]}T{index:02d}',
                              'SPOTPrice': value['Value']}
                processed_data.append(data_point)

        # Quick sanity check here
        if not self.check_data(processed_data):
            processed_data = self.repair_data(processed_data)

        return processed_data

    def check_data(self, data: list) -> bool:
        """
        Checks if the data was downloaded correctly.
        :param data: The downloaded data
        :return: bool True if download was successful, False otherwise
        """
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        # Check that no duplicates
        df = pd.DataFrame(data=data)
        all_times = list(df.Time)
        assert len(all_times) == len(set(all_times))

        if self.end_date != 'latest':
            return len(data) == self._get_num_days() * 24
        else:
            now = datetime.now()
            end = datetime.strptime(now.strftime('%Y-%m-%d'), '%Y-%m-%d')
            last_day = datetime.strptime(data[-1]['Time'][:10], '%Y-%m-%d')
            last_time = int(data[-1]['Time'][11:])
            days_missing = (end - last_day).days
            return len(data) == (self._get_num_days() - days_missing) * 24 \
                - 23 + last_time

    def repair_data(self, data: list) -> list:
        """
        There was an issue where one date did not have data in the upstream
        dataset provided by montel.
        This function fixes this problem by copying the data from the previous
        day to ensure a data series without missing entries.
        :param data: The data with missing entries as list of dicts
        :return: The original list of dicts with added missing entries
        """
        df = pd.DataFrame(data=data)
        all_times = list(df.Time)
        missing_dates = []
        complete_days = self._get_num_days()
        if self.end_date == 'latest':
            now = datetime.now()
            end = datetime.strptime(now.strftime('%Y-%m-%d'), '%Y-%m-%d')
            last_day = datetime.strptime(data[-1]['Time'][:10], '%Y-%m-%d')
            days_missing = (end - last_day).days
            complete_days -= (days_missing + 1)
        for i in range(complete_days):
            for h in range(24):
                time_str = (datetime.strptime(self.start_date, '%Y-%m-%d') +
                            timedelta(days=i, hours=h)).strftime('%Y-%m-%dT%H')
                if not time_str in all_times:
                    missing_dates.append(time_str[:10])
                    start_index = i * 24 + h
                    data.insert(start_index, data[start_index - 24].copy())
                    data[start_index]['Time'] = time_str
        print(f'Repaired missing montel data from dates: '
              f'{np.unique(missing_dates)}')
        if self.start_date in missing_dates:
            raise RuntimeError('The start date you picked is too soon! '
                               'No data available for montel API.')
        assert self.check_data(data)
        return data


class BearerTokenMissingError(Exception):
    def __init__(self):
        super().__init__('The Bearer Token for the Montel API could not be '
                         'found or is invalid. Please store the token in '
                         'the MONTEL_TOKEN environment variable or connect '
                         'to the MWN to retrieve the token automatically.')


if __name__ == '__main__':
    dg = MontelDataGetter()
    dg.get_data('2021-08-01', 'latest')
