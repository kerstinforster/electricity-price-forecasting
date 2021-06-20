""" This class implements a data getter that downloads the montel electricity
stock market price data"""

import requests
import json
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
        self.token = 'VHwQF502L7rCkl-URyn0BvqzkFPZ_V9ovU5MOXYsDaUYbjVBGz37jJUx7Kgj01BEOxU6w3b8c6uxUWCEpXvQSZR_hKvL5ObPKNKAaV7a_iWmLPaa8aQmm9Whu2pK0sfa-e5pb_4R8XlMN8GAl8kci8nFWyR9bN2nBNR459ogXsHgoINsi142fQBHXWj7TVx7kKKFYuAwNiNNwXJfz-2q2DcNATrx6cco_QXkj2az7Tz0mBVaM-cWyN8ykWphitzWjrqEbhUEi73Mob1jNkrhfOTlqvErctozYeYvBFK7MON6Hc6paPPqr0FVByeFuU9kkUT-sT_DgIlbBpZWHqiu8rPeYgsC-sMxPpTAusKAhGjYa7oo4WvSH-5OLGnY3X0rzKP0ojFAasOhwFo7570oswt-AgtdISOf2c7LkTRDaJw'  # pylint: disable=C0301
        self._token_check()

    def _token_check(self) -> None:
        """
        This function checks whether the token is still valid
        :raise PermissionError: if Bearer Token invalid
        """
        response = requests.get(
            'https://api.montelnews.com/spot/getmetadata',
            headers={'Authorization': f'Bearer {self.token}'})
        if response.status_code != 200:
            raise PermissionError(f'The MontelBearer Token seems to be invalid,'
                                  f' status {response.status_code}, \nresponse:'
                                  f' {response.text} \n Most likely, you need'
                                  f' to get the new key from moodle.tum.de')

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
            'toDate': self.end_date,  # yyyy-mm-dd
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
        for element in data_el:
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
                data_point = dict()
                data_point['Time'] = \
                    f'{element["Date"]}T{index:02d}'
                data_point['Value'] = value['Value']
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
        return len(data) == self._get_num_days() * 24

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
        all_times = np.unique(list(df.Time))
        missing_dates = []
        for i in range(self._get_num_days()):
            for h in range(24):
                time_str = (datetime.strptime(self.start_date, '%Y-%m-%d') +
                            timedelta(days=i, hours=h)).strftime('%Y-%m-%dT%H')
                if not time_str in all_times:
                    missing_dates.append(time_str[:10])
                    start_index = i * 24 + h
                    data.insert(start_index, data[start_index - 24])
                    data[start_index]['Time'] = time_str
        print(f'Repaired missing montel data from dates: '
              f'{np.unique(missing_dates)}')
        if self.start_date in missing_dates:
            raise RuntimeError('The start date you picked is too soon! '
                               'No data available for montel API.')
        assert self.check_data(data)
        return data


if __name__ == '__main__':
    dg = MontelDataGetter()
    dg.get_data()
