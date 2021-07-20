""" This class implements a data getter that downloads the weather data"""
import os
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from geopy.geocoders import Nominatim
from pathlib import Path
from src.data.data_getter import BaseDataGetter


class WeatherDataGetter(BaseDataGetter):
    """
    This class is responsible for downloading the Weather data.
    This data contains Temperature(°C), Precipitation(mm), Wind Speed(km/h),
    Humidity(%) and Atmospheric Pressure(hPa) in an hourly interval for a specific location.
    pip """
    def __init__(self, name="weather", location="Munich, Germany"):
        """
        Constructor for the Weather Data Getter
        """
        super().__init__(name)
        geolocator = Nominatim(user_agent="weather_agent")      
        self.location = location
        self.latitude = geolocator.geocode(location).latitude
        self.longitude = geolocator.geocode(location).longitude
        self.now_date = datetime.now().strftime('%Y-%m-%dT%H')

    def check_data_coverage(self, data):
        """
        return Integer between 0 (no records) and 1 (all records)
        """
        print("Data coverage {0:.0%}".format(data.coverage()))

    def _get_raw_data(self):
        """
        This function queries hourly weather data for
        a single geographical point.
        :return: meteostat.interface.hourly.Hourly object
        """
        position = Point(self.latitude, self.longitude)

        # Get all the data point until the last hour of the last day        
        end_date = self.now_date if self.end_date == "latest" \
            else self.end_date + self.end_time
        data = Hourly(loc=position,
                      start=datetime.strptime(self.start_date, "%Y-%m-%d"),
                      end=datetime.strptime(end_date, "%Y-%m-%dT%H"))
        # Check data coverage
        self.check_data_coverage(data)

        return data

    def _process_raw_data(self, data):
        """
        This function gives access to the resulting DataFrame of the query
        :return: Pandas DataFrame
        """
        fp_data = data.fetch()
        fp_data.index.name = "Time"
        fp_data.reset_index(level=0, inplace=True)

        return fp_data


if __name__ == "__main__":
    wg = WeatherDataGetter()
    wg.get_data()
