""" This class implements a data getter that downloads the weather data"""
import os
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from geopy.geocoders import Nominatim
from pathlib import Path
from data_getter import BaseDataGetter


class WeatherDataGetter(BaseDataGetter):
    """
    This class is responsible for downloading the Weather data.
    This data contains Temperature(°C), Precipitation(mm), Wind Speed(km/h),
    Humidity(%) and Atmospheric Pressure(hPa) in an hourly interval for a specific location.    
    """
    def __init__(self, name="weather", location="Munich, Germany"):
        """
        Constructor for the Weather Data Getter
        """
        BaseDataGetter.__init__(self, name=name)
        geolocator = Nominatim(user_agent="weather_agent")      
        self.location = location
        self.latitude = geolocator.geocode(location).latitude
        self.longitude = geolocator.geocode(location).longitude

    

    def get_weather_data(self, start_date='2012-01-01', end_date='2021-06-21'):       
        """
        fetches the data from meteostat
        date must be in format 'YYYY-MM-DD'        
        return DataFrame        
        """
        self.start_date = start_date         
        self.end_date = end_date
        # self.timezone = timezone
        position = Point(self.latitude, self.longitude)
        # Get all the data point until the last hour of the last day
        end_date = end_date + ' ' + '23:59'                            
        data = Hourly(loc=position, start=datetime.strptime(start_date, '%Y-%m-%d'), \
            end=datetime.strptime(end_date, '%Y-%m-%d %H:%M'))
        # Check data coverage
        self.check_data_coverage(data)        
        # Access the resulting DataFrame
        data = data.fetch()
        
        return data

    def save_weather_data(self, data, overwrite=False):        
        """
        Saves the dataframe to a csv file in 
        """
        if overwrite or not os.path.exists(os.path.join(self.data_dir, 'weather_data.csv')):
            # Save DataFrame to csv
            data.to_csv(os.path.join(self.data_dir, 'weather_data.csv'))
        
        return       

    def check_data_coverage(self, data):
        """
        return Integer between 0 (no records) and 1 (all records)
        """
        return print("Data coverage of this period is   {0:.0%}".format(data.coverage()))       
    

    def _get_raw_data():
        """
        Abtract function from base class not needed here
        """
        raise NotImplementedError

    def _process_raw_data():
        """
        Abtract function from base class not needed here
        """
        raise NotImplementedError
        


   
if __name__ == '__main__':
    weatherguy = WeatherDataGetter()
    df = weatherguy.get_weather_data()
    weatherguy.save_weather_data(data=df)