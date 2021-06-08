""" Validate that the montel data getter works as expected """

import pytest
import pandas as pd

from src.data.montel_data_getter import *


def test_init():
    dg = MontelDataGetter("montel_test")
    path = dg.get_project_root()
    assert os.path.exists(
        os.path.join(path, 'data', 'montel_test')
    )
    assert dg.data_dir == os.path.join(path, 'data', 'montel_test')
    assert dg.name == "montel_test"
    assert dg._root_dir == path
    os.removedirs(dg.data_dir)


def test_show_all_datasets():
    dg = MontelDataGetter("montel_test")
    dg._show_available_datasets()
    os.removedirs(dg.data_dir)


def test_get_and_process_data():
    dg = MontelDataGetter("montel_test")

    dg.get_data('2020-01-01', '2020-12-31')

    assert os.path.exists(os.path.join(dg.data_dir, 'data.csv'))

    df = pd.read_csv(os.path.join(dg.data_dir, 'data.csv'))
    df.head()
    assert df.shape == (366, 27)  # leap year, 24hours + 3 fields

    os.remove(os.path.join(dg.data_dir, 'data.csv'))
    os.removedirs(dg.data_dir)


def test_wrong_token():
    dg = MontelDataGetter("montel_test")
    dg.token = "wrong_token"
    with pytest.raises(PermissionError):
        dg._token_check()