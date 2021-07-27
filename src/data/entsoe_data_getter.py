""" This class implements a data getter that downloads the entso-e data"""

import pandas as pd
from datetime import datetime
import sys, time
from entsoe import EntsoePandasClient

from src.data.data_getter import BaseDataGetter


class EntsoeDataGetter(BaseDataGetter):
    """
    This class is responsible for downloading the entso-e data.
    This data contains "PSRTYPE_MAPPINGS" in an hourly interval.
    """

    def __init__(self, name: str = 'entsoe'):
        """
        Constructor for the entso-e Data Getter
        """
        super().__init__(name)
        self.client = EntsoePandasClient(api_key="a98acd60-5cc2-4208-a0a8-cd4700ab41ba", retry_count=3,retry_delay=2)
        self.country_code1 = 'DE_AT_LU'
        self.country_code2 = 'DE_LU'
        self.PSRTYPE_MAPPINGS = {
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

    def check_data_coverage(self, data):
        """
        return Integer between 0 (no records) and 1 (all records)
        """
        print('Data coverage {0:.0%}'.format(data.coverage()))

    def _time_split(self, start, end):
        start1, end1, start2, end2 = None, None, None, None
        split1 = pd.Timestamp('20181001', tz='Europe/Brussels')
        split2 = pd.Timestamp('20181001', tz='Europe/Brussels')
        
        if start <= split1 and end <= split1:
            start1, end1 = start, end
        elif start >= split2 and end >= split2:
            start2, end2 = start, end
        elif start <= split1 and end >= split2:
            start1, end1, start2, end2 = start, split1, split2, end

        return start1,end1,start2,end2

    def _get_generation_data(self, psr_type_str, name_str, start, end, pop_column=2 ,pop_column_tuple=None):
        """
        return generation data of PSRTYPE_MAPPINGS
        :return: Pandas DataFrame
        """
        start1,end1,start2,end2 = self._time_split(start, end)

        print('Get "' + name_str + '" data...')
        start_time = time.time()

        gen_c1 = {} 
        gen_c2 = {} 
        gen_temp = {}
        
        if start1 != None and end1 != None:
            gen_c1 = self.client.query_generation(self.country_code1, start=start1,end=end1, psr_type=psr_type_str).rename_axis('time').resample('1H').sum()/4
            try:
                gen_c1.pop(pop_column_tuple)
            except: {}
            try:
                gen_c1.set_axis(['val1', 'val2'], axis='columns', inplace=True)
                gen_c1['gen_c1'] = (gen_c1['val1'] + gen_c1['val2']).astype("float64")
            except:  
                gen_c1.set_axis(['gen_c1'], axis='columns', inplace=True)
            gen_temp = gen_c1

        if start2 != None and end2 != None:
            gen_c2 = self.client.query_generation(self.country_code2, start=start2,end=end2, psr_type=psr_type_str).rename_axis('time').resample('1H').sum()/4    
            try:
                gen_c2.pop(pop_column_tuple)
            except: {}
            try:
                gen_c2.set_axis(['val1', 'val2'], axis='columns', inplace=True)
                gen_c2['gen_c2'] = (gen_c2['val1'] + gen_c2['val2']).astype("float64")
            except:
                gen_c2.set_axis(['gen_c2'], axis='columns', inplace=True)
            gen_c2 = gen_c2[['gen_c2']]
            gen_temp = gen_c2

        if start1 != None and end1 != None and start2 != None and end2 != None:
            gen = pd.merge(gen_c1, gen_c2, left_on='time', right_on='time', how='outer')
            gen = gen.fillna(0)
            gen[name_str] = (gen['gen_c1'] + gen['gen_c2']).astype("float64")
        elif start1 != None and end1 != None:
            gen = gen_temp
            gen = gen.fillna(0)
            gen[name_str] = (gen['gen_c1']).astype("float64")
        elif start2 != None and end2 != None:
            gen = gen_temp
            gen = gen.fillna(0)
            gen[name_str] = (gen['gen_c2']).astype("float64")
        
        gen = gen[[name_str]]


        print('Time: '+ str(time.time() - start_time))
        return gen

    def _get_load_data(self, start, end):
        """
        return load data
        :return: Pandas DataFrame
        """
        print('Get "Load" data...')
        start_time = time.time()
        start1,end1,start2,end2 = self._time_split(start, end)

        load1 = {}
        load2 = {}
        load_temp = {}

        if start1 != None and end1 != None:
            load1 = self.client.query_load(self.country_code1, start=start1,end=end1).to_frame("Load").rename_axis('time').resample('1H').sum()/4
            load1.set_axis(['load1'], axis='columns', inplace=True)
            if start2 != None and end2 != None:
                load1.append(pd.DataFrame({'time': pd.Timestamp('201810010000', tz='Europe/Brussels'), 'load1': load1.load1[pd.Timestamp('201809302300', tz='Europe/Brussels')]}))
            load_temp = load1

        if start2 != None and end2 != None:
            load2 = self.client.query_load(self.country_code2, start=start2,end=end2).to_frame("Load").rename_axis('time').resample('1H').sum()/4
            load2.set_axis(['load2'], axis='columns', inplace=True)
            load_temp = load2

        if start1 != None and end1 != None and start2 != None and end2 != None:
            load = pd.merge(load1, load2, left_on='time', right_on='time', how='outer')
            load = load.fillna(0)
            load['Load'] = (load['load1'] + load['load2']).astype("float64")
        elif start1 != None and end1 != None:
            load= load_temp
            load = load.fillna(0)
            load['Load'] = (load['load1']).astype("float64")
        elif start2 != None and end2 != None:
            load= load_temp
            load = load.fillna(0)
            load['Load'] = (load['load2']).astype("float64")

        load = load[['Load']]

        print('Time: '+ str(time.time() - start_time))
        return load

    def _get_raw_data(self):
        """
        This function queries hourly Entso-E data.
        :return: Pandas DataFrame
        """
        data = self._get_load_data(self.start_date, self.end_date)
        for key in self.PSRTYPE_MAPPINGS:
            data_temp = self._get_generation_data(key, self.PSRTYPE_MAPPINGS[key], self.start_date, self.end_date, 2, (self.PSRTYPE_MAPPINGS[key], 'Actual Consumption'))
            data = pd.merge(data, data_temp, left_on='time', right_on='time')

        return data

    def _process_raw_data(self, data):
        """
        This function gives access to the resulting DataFrame of the query
        :return: Pandas DataFrame
        """
        fp_data = data.fetch()
        fp_data.index.name = 'Time'
        #if not self.check_data(fp_data):
        #    raise ValueError('Entso-E Data is corrupted!')

        return fp_data

    def check_data(self, data: pd.DataFrame) -> bool:
        """
        Checks if the data was downloaded correctly.
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
    wg =EntsoeDataGetter()
    wg.get_data()
