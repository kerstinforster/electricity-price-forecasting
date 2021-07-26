import os
import pandas as pd
from datetime import datetime

from src.data.weather_data_getter import WeatherDataGetter


def test_init():
    wg = WeatherDataGetter("weather_test")
    path = wg.get_project_root()
    assert os.path.exists(
        os.path.join(path, 'data', 'weather_test')
    )
    assert wg.data_dir == os.path.join(path, 'data', 'weather_test')
    assert wg.name == "weather_test"
    assert wg._root_dir == path
    if os.path.exists(os.path.join(wg.data_dir, 'data.csv')):
        os.remove(os.path.join(wg.data_dir, 'data.csv'))
    os.removedirs(wg.data_dir)


def test_get_and_process_data():
    wg = WeatherDataGetter("weather_test")

    wg.get_data('2020-01-01', '2020-12-31', overwrite=True)

    assert os.path.exists(os.path.join(wg.data_dir, 'data.csv'))

    df = pd.read_csv(os.path.join(wg.data_dir, 'data.csv'))
    df.head()
    assert df.shape == (wg._get_num_days() * 24, 8)

    os.remove(os.path.join(wg.data_dir, 'data.csv'))
    os.removedirs(wg.data_dir)


def test_get_and_process_data_latest():
    wg = WeatherDataGetter("weather_test")

    wg.get_data('2021-01-01', 'latest', overwrite=True)

    assert os.path.exists(os.path.join(wg.data_dir, 'data.csv'))

    df = pd.read_csv(os.path.join(wg.data_dir, 'data.csv'))
    df.head()
    assert df.shape == (wg._get_num_days() * 24 - 23 + datetime.now().hour, 8)

    os.remove(os.path.join(wg.data_dir, 'data.csv'))
    os.removedirs(wg.data_dir)


