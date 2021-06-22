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
    def __init__(self, name="weather", city="Munich, Germany"):
        """
        Constructor for the Weather Data Getter
        """
        BaseDataGetter.__init__(self, name=name)
        geolocator = Nominatim(user_agent="weather_agent")      
        self.location = city
        self.latitude = geolocator.geocode(city).latitude
        self.longitude = geolocator.geocode(city).longitude        
    

    def get_weather_data(self, start_date='2012-01-01', end_date='2021-06-01', overwrite=False):       
        """
        fetches the data from meteostat
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')         
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        position = Point(self.latitude, self.longitude)                            
        data = Hourly(position, self.start_date, self.end_date)
        data = data.fetch()
        
        return data

    def save_weather_data(self, data, overwrite=False):        
        """
        Saves the dataframe to a csv file
        """
        if overwrite or not os.path.exists(os.path.join(self.data_dir, 'weather_data.csv')):
            # Save DataFrame to csv
            data.to_csv(os.path.join(self.data_dir, 'weather_data.csv'))
        
        return

    def _get_raw_data():
        raise NotImplementedError

    def _process_raw_data():
        raise NotImplementedError
        


   
if __name__ == '__main__':
    weatherguy = WeatherDataGetter()
    df = weatherguy.get_weather_data(start_date='2020-01-01')
    weatherguy.save_weather_data(data=df)