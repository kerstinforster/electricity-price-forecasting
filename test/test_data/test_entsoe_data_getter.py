import pytest
import os
import pandas as pd
from datetime import datetime

from src.data.entsoe_data_getter import EntsoeDataGetter


def test_init():
    eg = EntsoeDataGetter("entsoe_test")
    path = eg.get_project_root()
    assert os.path.exists(
        os.path.join(path, 'data', 'entsoe_test')
    )
    assert eg.data_dir == os.path.join(path, 'data', 'entsoe_test')
    assert eg.name == "entsoe_test"
    assert eg._root_dir == path
    if os.path.exists(os.path.join(eg.data_dir, 'data.csv')):
        os.remove(os.path.join(eg.data_dir, 'data.csv'))
    os.removedirs(eg.data_dir)


def test_get_and_process_data():
    eg = EntsoeDataGetter("entsoe_test")

    eg.get_data('2020-01-01', '2020-12-31', overwrite=True)

    assert os.path.exists(os.path.join(eg.data_dir, 'data.csv'))

    df = pd.read_csv(os.path.join(eg.data_dir, 'data.csv'))
    df.head()
    assert df.shape == (eg._get_num_days() * 24, 5)

    # Check that error is thrown on too old start date
    with pytest.raises(ValueError):
        eg.get_data('2014-01-01', '2014-02-01', 'T12')

    os.remove(os.path.join(eg.data_dir, 'data.csv'))
    os.removedirs(eg.data_dir)


def test_get_and_process_data_latest():
    eg = EntsoeDataGetter("entsoe_test")

    eg.get_data('2021-08-01', 'latest', overwrite=True)

    assert os.path.exists(os.path.join(eg.data_dir, 'data.csv'))

    df = pd.read_csv(os.path.join(eg.data_dir, 'data.csv'))
    df.head()
    assert df.shape == (eg._get_num_days() * 24 - 23 + datetime.now().hour, 5)

    os.remove(os.path.join(eg.data_dir, 'data.csv'))
    os.removedirs(eg.data_dir)