""" This class implements a data getter that downloads the entso-e data"""

import pandas as pd
# import time
import numpy as np
from typing import Any
from datetime import datetime, timedelta
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
        start = pd.Timestamp(start, tz='Europe/Berlin')
        end = pd.Timestamp(end, tz='Europe/Berlin') + pd.DateOffset(1)
        start1, end1, start2, end2 = None, None, None, None
        split1 = pd.Timestamp('20181001', tz='Europe/Berlin')
        split2 = pd.Timestamp('20181001', tz='Europe/Berlin')

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
        This function ignores the end_time parameter and just tries to get all
        data for the end_date. The processing function deals with this issue.
        :return: Dataframe containing the downloaded data
        """
        if self.end_date == 'latest':
            self.end_date = datetime.now().strftime('%Y-%m-%d')
            self.end_time = datetime.now().strftime('T%H')

        raw_data = self._get_load_data(self.start_date, self.end_date)
        for key, value in self.field_name_map.items():
            data_temp = self._get_generation_data(key,
                                                  value,
                                                  self.start_date,
                                                  self.end_date,
                                                  (value,
                                                   'Actual Consumption'))
            raw_data = pd.merge(raw_data, data_temp,
                                left_on='time', right_on='time')

        return raw_data

    def _process_raw_data(self, raw_data: Any):
        """
        This function gives access to the resulting DataFrame of the query
        :param raw_data: Dataframe that was generated by this data getter
        :return: Processed dataframe that contains the required data
        """
        # Fix time to not use time zones anymore
        raw_data = raw_data.rename(columns={'time': 'Time'})
        old_time = raw_data['Time'][0].tz_localize(None)
        raw_data['Time'] = raw_data['Time'].dt.tz_convert(None)
        time_delta = raw_data['Time'][0] - old_time
        raw_data['Time'] -= time_delta

        # Fix nan values
        raw_data = raw_data.fillna(method='ffill')

        # Fix end_time setting
        if pd.Timestamp(raw_data['Time'].values[-1]).strftime('%Y-%m-%d') \
                == self.end_date:
            current_end_time = pd.DatetimeIndex(
                [raw_data['Time'].values[-1]]).hour[0]
            wanted_end_time = int(self.end_time[-2:])
            if current_end_time != wanted_end_time:
                raw_data = \
                    raw_data.iloc[:-(current_end_time - wanted_end_time), :]
            assert pd.DatetimeIndex([raw_data['Time'].values[-1]]).hour[0] \
                   == wanted_end_time

        # Fix entries missing
        if not self.check_data(raw_data):
            # Data is missing because API not up to date -> Need to fix it
            raw_data = self.repair_data(raw_data)

        return raw_data

    def check_data(self, df: pd.DataFrame) -> bool:
        """
        Checks if the data is complete and no values are missing.
        :param df: The downloaded data
        :return: bool True if download was successful, False otherwise
        """

        all_times = list(df.Time)
        assert len(all_times) == len(set(all_times))

        if self.end_date != 'latest':
            expected_length = self._get_num_days() * 24 \
                - 23 + int(self.end_time[-2:])
        else:
            expected_length = (self._get_num_days()) * 24 \
                - 23 + int(self.now_date[-2:])
        return expected_length == len(df)

    def repair_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data_list = df.to_dict('records')
        all_times = list(df.Time)
        all_times = [a.strftime('%Y-%m-%dT%H') for a in all_times]
        missing_dates = []
        complete_days = self._get_num_days()
        if self.end_date == 'latest':
            now = datetime.now()
            end = datetime.strptime(now.strftime('%Y-%m-%d'), '%Y-%m-%d')
            last_day = datetime.strptime(df[-1]['Time'][:10], '%Y-%m-%d')
            days_missing = (end - last_day).days
            complete_days -= (days_missing + 1)

        for i in range(complete_days):
            for h in range(24):
                time_str = pd.Timestamp(datetime.strptime(self.start_date,
                                                          '%Y-%m-%d') +
                            timedelta(days=i, hours=h)).strftime('%Y-%m-%dT%H')
                if not time_str in all_times:
                    missing_dates.append(time_str[:10])
                    start_index = i * 24 + h
                    data_list.insert(start_index,
                                     data_list[start_index - 24].copy())
                    data_list[start_index]['Time'] = time_str

        print(f'Repaired missing entsoe data from dates: '
              f'{np.unique(missing_dates)}')
        if self.start_date in missing_dates:
            raise RuntimeError('The start date you picked is too soon! '
                               'No data available for entsoe API.')
        df = pd.DataFrame(data=data_list)

        # Fix too many items now
        current_end_time = pd.DatetimeIndex([df['Time'].values[-1]]).hour[0]
        wanted_end_time = int(self.end_time[-2:])
        if current_end_time != wanted_end_time:
            df = df.iloc[:-(current_end_time - wanted_end_time), :]
        assert self.check_data(df)
        return df


if __name__ == '__main__':
    eg = EntsoeDataGetter()
    data = eg.get_data(start_date='2021-01-01',
                       end_date='latest', end_time='T16')
    print(data.tail())
