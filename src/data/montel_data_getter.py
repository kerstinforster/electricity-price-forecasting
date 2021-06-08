""" This class implements a data getter that downloads the montel electricity
stock market price data"""

import requests
import os
import json
import pandas as pd
from typing import Any

from src.data.data_getter import DataGetter


class MontelDataGetter(DataGetter):
    """
    This class is responsible for downloading the Montel dataset(s)
    """
    def __init__(self, name: str = 'montel'):
        """
        Constructor for the Montel Data Getter
        """
        super().__init__(name)
        self.token = 'Zeo4XP4SswsOwnradSkZQ6Xh16eEi10UNtTE1Q3ek4lS2Xv0Nx40RKs' \
                     'KqxryzUCwAV6fzmt2erLFigT9iNy_IYcq85A7jO-kE7mRun8Dpk6BH6' \
                     'xc0mMVzzyog8ZnK3Jk3X_8drbMbGqMUeFGc7ul06CERBN_QZ4ySQdOR' \
                     '1EAYOLzVyHkae7d3KBdxLazP3QvYohPhIYGScKmNipVkhOpTdQwROce' \
                     'ZS54CGCw8QSWm8wv5vnCNXYxAl7fxd4nEZEaZhHqOgKMnw2MgjtHYJ2' \
                     'jkhfnsOQcbP52zp-6wI6-YyXWoiOphouS7w10Le0kacBQfDXVzMjoGS' \
                     'FIdPCRxB4qTrXT6n14OO2VwdrXVO4wxbYN65fUYWntRE02d820pw4fE' \
                     'rykZaUcQFCTEqWC9XHU_FWlSC2yZAGNrMFBrh5EiZ4'
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

    def _get_raw_data(self, start_date: str = '2012-01-01',
                  end_date: str = '2021-06-01') -> Any:
        """
        This function downloads the datasets that shall be used for training.
        :return: json data from the API
        """
        params = {
            'spotKey': 14,  # This is the key we should use
            'fields': ['Base', 'Peak', 'Hours'],
            'fromDate': start_date,  # yyyy-mm-dd
            'toDate': end_date,  # yyyy-mm-dd
            'currency': 'eur',
            'sortType': 'ascending'
        }
        response = requests.get(
            'https://api.montelnews.com/spot/getprices',
            headers={'Authorization': f'Bearer {self.token}'},
            params=params
        )
        return response.json()

    @staticmethod
    def _process_raw_data(data: Any) -> Any:
        """
        Data preprocessing method that transforms the json response into more
        useful and quicker-to-access data
        :param data: The raw json data from the API
        :return: list of dicts that is suitable for pandas
        """
        data_el = data['Elements']
        for element in data_el:
            element['Date'] = element['Date'][:10]

            # For the changes from summer to winter time we don't have 24 hours
            if len(element['TimeSpans']) == 23:
                element['TimeSpans'].append(element['TimeSpans'][-1])
            elif len(element['TimeSpans']) == 25:
                element['TimeSpans'] = element['TimeSpans'][:24]

            for index, value in enumerate(element['TimeSpans']):
                element[f't{index}'] = value['Value']
            del element['TimeSpans']
        return data_el

    def get_data(self, start_date: str = '2012-01-01',
                 end_date: str = '2021-06-01') -> None:
        """
        Main method that gets the montel EEX data and stores it in a csv file
        :param start_date: Start date for the data to get in format yyyy-mm-dd
        :param end_date: End date for the data to get in format yyyy-mm-dd
        """
        data = self._get_raw_data(start_date, end_date)
        print(f'Received {data["SpotName"]} spot data.')
        processed_data = self._process_raw_data(data)
        df = pd.DataFrame(data=processed_data)
        df.to_csv(os.path.join(self.data_dir, 'data.csv'), index=False)


if __name__ == '__main__':
    dg = MontelDataGetter()
    dg.get_data()
