""" This class implements a data getter that downloads the entso-e data"""

import pandas as pd
# import time
from typing import Any
from datetime import datetime
from entsoe import EntsoePandasClient

from src.data.data_getter import BaseDataGetter


class EntsoeDataGetter(BaseDataGetter):
    """
    This class is responsible for downloading the entso-e data.
    This data contains load data and electricity generation data in an hourly
    interval. Fields included are Solar, Wind Offshore / Onshore and Load data.
    """

    def __init__(self, name: str = 'entsoe'):
        """
        Constructor for the entso-e Data Getter
        :param name: Data Getter name
        """
        super().__init__(name)
        self.api_key = 'a98acd60-5cc2-4208-a0a8-cd4700ab41ba'
        self.client = EntsoePandasClient(api_key=self.api_key, retry_count=3,
                                         retry_delay=2)
        self.now_date = datetime.now().strftime('%Y-%m-%dT%H')
        self.country_code1 = 'DE_AT_LU'
        self.country_code2 = 'DE_LU'
        self.field_name_map = {
            # 'A03': 'Mixed',
            # 'A04': 'Generation',
            # 'A05': 'Load',
            # 'B01': 'Biomass',
            # 'B02': 'Fossil Brown coal/Lignite',
            # 'B03': 'Fossil Coal-derived gas',
            # 'B04': 'Fossil Gas',
            # 'B05': 'Fossil Hard coal',
            # 'B06': 'Fossil Oil',
            # 'B07': 'Fossil Oil shale',
            # 'B08': 'Fossil Peat',
            # 'B09': 'Geothermal',
            # 'B10': 'Hydro Pumped Storage',
            # 'B11': 'Hydro Run-of-river and poundage',
            # 'B12': 'Hydro Water Reservoir',
            # 'B13': 'Marine',
            # 'B14': 'Nuclear',
            # 'B15': 'Other renewable',
            'B16': 'Solar',
            # 'B17': 'Waste',
            'B18': 'Wind Offshore',
            'B19': 'Wind Onshore',
            # 'B20': 'Other',
            # 'B21': 'AC Link',
            # 'B22': 'DC Link',
            # 'B23': 'Substation',
            # 'B24': 'Transformer'
        }

    def _time_split(self, start: str, end: str) -> tuple:
        """
        Split the time into dates before first of october 2018 and after.
        The reason for that is that the country code name was changed at that
        date.
        :param start: The start date string
        :param end: The end date string
        :return: Tuple that contains start and end date for both splits
        """
        start = pd.Timestamp(start, tz='Europe/Brussels')
        end = pd.Timestamp(end, tz='Europe/Brussels') + pd.DateOffset(1)
        start1, end1, start2, end2 = None, None, None, None
        split1 = pd.Timestamp('20181001', tz='Europe/Brussels')
        split2 = pd.Timestamp('20181001', tz='Europe/Brussels')

        if start <= split1 and end <= split1:
            start1, end1 = start, end
        elif start >= split2 and end >= split2:
            start2, end2 = start, end
        elif start <= split1 and end >= split2:
            start1, end1, start2, end2 = start, split1, split2, end

        return start1, end1, start2, end2

    def _get_generation_data(self, psr_type_str: str, name_str: str, start: str,
                             end: str, pop_column_tuple: tuple = None) \
            -> pd.DataFrame:
        """
        Get the electricity generation data fields
        :param psr_type_str: The string of the field in the API
        :param name_str: The name of the generation source
        :param start: The start date
        :param end: The end date
        :param pop_column_tuple: The tuple (MultiIndex) of the column to drop
        :return: DataFrame that contains the generation data
        """
        start1, end1, start2, end2 = self._time_split(start, end)
        # print(f'Getting {name_str} data')
        # start_time = time.time()

        gen = None
        gen_c1 = {}
        gen_c2 = {}
        gen_temp = {}

        if start1 is not None and end1 is not None:
            gen_c1 = self.client.query_generation(self.country_code1,
                                                  start=start1,
                                                  end=end1,
                                                  psr_type=psr_type_str)\
                         .rename_axis('time').resample('1H').sum()/4
            if pop_column_tuple in gen_c1.columns:
                gen_c1.pop(pop_column_tuple)
            try:
                gen_c1.set_axis(['val1', 'val2'], axis='columns', inplace=True)
                gen_c1['gen_c1'] = (gen_c1['val1'] + gen_c1['val2']).astype(
                    'float64')
            except ValueError:
                gen_c1.set_axis(['gen_c1'], axis='columns', inplace=True)
            gen_temp = gen_c1

        if start2 is not None and end2 is not None:
            gen_c2 = self.client.query_generation(self.country_code2,
                                                  start=start2,
                                                  end=end2,
                                                  psr_type=psr_type_str)\
                         .rename_axis('time').resample('1H').sum()/4
            if pop_column_tuple in gen_c2.columns:
                gen_c2.pop(pop_column_tuple)

            try:
                gen_c2.set_axis(['val1', 'val2'], axis='columns', inplace=True)
                gen_c2['gen_c2'] = (gen_c2['val1'] + gen_c2['val2']).astype(
                    'float64')
            except ValueError:
                gen_c2.set_axis(['gen_c2'], axis='columns', inplace=True)
            gen_c2 = gen_c2[['gen_c2']]
            gen_temp = gen_c2

        if start1 is not None and end1 is not None and start2 is not None and \
                end2 is not None:
            gen = pd.merge(gen_c1, gen_c2, left_on='time', right_on='time',
                           how='outer')
            gen = gen.fillna(0)
            gen[name_str] = (gen['gen_c1'] + gen['gen_c2']).astype('float64')
        elif start1 is not None and end1 is not None:
            gen = gen_temp
            gen = gen.fillna(0)
            gen[name_str] = (gen['gen_c1']).astype('float64')
        elif start2 is not None and end2 is not None:
            gen = gen_temp
            gen = gen.fillna(0)
            gen[name_str] = (gen['gen_c2']).astype('float64')

        gen.reset_index(level=0, inplace=True)
        gen = gen[['time', name_str]]

        #print(f'{name_str} Time: {str(time.time() - start_time)}')
        return gen

    def _get_load_data(self, start: str, end: str):
        """
        Get load data from the API
        :param start: String that contains the start date
        :param end: String that contains the end date
        :return: Dataframe that contains the load data
        """
        # start_time = time.time()
        start1, end1, start2, end2 = self._time_split(start, end)

        load = None
        load1 = {}
        load2 = {}
        load_temp = {}

        if start1 is not None and end1 is not None:
            load1 = self.client.query_load(self.country_code1, start=start1,
                                           end=end1).to_frame('Load') \
                        .rename_axis('time').resample('1H').sum()/4
            load1.set_axis(['load1'], axis='columns', inplace=True)
            load1.reset_index(level=0, inplace=True)
            if start2 is not None and end2 is not None:
                load1 = load1.append(pd.DataFrame(
                    {'time': [pd.Timestamp('201810010000', tz='Europe/Berlin')],
                     'load1': [load1.load1.iloc[-1]]}))
            load_temp = load1

        if start2 is not None and end2 is not None:
            load2 = self.client.query_load(self.country_code2, start=start2,
                                           end=end2).to_frame('Load')\
                        .rename_axis('time').resample('1H').sum()/4
            load2.set_axis(['load2'], axis='columns', inplace=True)
            load2.reset_index(level=0, inplace=True)
            load_temp = load2

        if start1 is not None and end1 is not None and start2 is not None and \
                end2 is not None:
            load = pd.merge(load1, load2, left_on='time', right_on='time',
                            how='outer')
            load = load.fillna(0)
            load['Load'] = (load['load1'] + load['load2']).astype('float64')
        elif start1 is not None and end1 is not None:
            load = load_temp
            load = load.fillna(0)
            load['Load'] = (load['load1']).astype('float64')
        elif start2 is not None and end2 is not None:
            load = load_temp
            load = load.fillna(0)
            load['Load'] = (load['load2']).astype('float64')

        load = load[['time', 'Load']]

        # print('Load Time: ' + str(time.time() - start_time))
        return load

    def _get_raw_data(self):
        """
        This function queries hourly Entso-E data.
        :return: Dataframe containing the downloaded data
        """
        data = self._get_load_data(self.start_date, self.end_date)
        for key in self.field_name_map:
            data_temp = self._get_generation_data(key,
                                                  self.field_name_map[key],
                                                  self.start_date,
                                                  self.end_date,
                                                  (self.field_name_map[key],
                                                   'Actual Consumption'))
            data = pd.merge(data, data_temp, left_on='time', right_on='time')

        return data

    def _process_raw_data(self, data: Any):
        """
        This function gives access to the resulting DataFrame of the query
        :param data: Dataframe that was generated by this data getter
        :return: Processed dataframe that contains the required data
        """
        data = data.rename(columns={'time': 'Time'})
        old_time = data['Time'][0].tz_localize(None)
        data['Time'] = data['Time'].dt.tz_convert(None)
        time_delta = data['Time'][0] - old_time
        data['Time'] -= time_delta

        print(data.head())
        if not self.check_data(data):
            raise ValueError('Entso-E Data is corrupted!')

        return data

    def check_data(self, data: pd.DataFrame) -> bool:
        """
        Checks if the data is complete and no values are missing.
        :param data: The downloaded data
        :return: bool True if download was successful, False otherwise
        """

        all_times = list(data.Time)
        assert len(all_times) == len(set(all_times))

        if self.end_date != 'latest':
            expected_length = self._get_num_days() * 24
        else:
            expected_length = (self._get_num_days()) * 24 \
                   - 23 + int(self.now_date[-2:])
        return expected_length == len(data)


if __name__ == '__main__':
    eg = EntsoeDataGetter()
    eg.get_data(start_date='2018-01-01', end_date='2018-12-31')
