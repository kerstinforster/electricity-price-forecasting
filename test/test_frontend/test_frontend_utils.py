""" Validate that the frontend utility functions work as expected """

import os
import pytest
from pathlib import Path

from src.frontend.frontend_utils import *
from src.data.montel_data_getter import MontelDataGetter

try:
    _ = MontelDataGetter()
    TOKEN_INVALID = False
except ConnectionRefusedError:
    TOKEN_INVALID = True


# Ugly test data fix cause this fkn shit montel token barely ever works:
def get_project_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture
def test_data():
    dg = MontelDataGetter()
    return dg.get_data(start_date='2016-01-01', end_date='2016-12-31')


# Tests:
def test_get_current_time():
    date, time = get_current_time()
    assert isinstance(date, str)
    assert isinstance(time, str)
    assert len(date) == 10  # YYYY-mm-dd
    assert len(time) == 5  # HH:MM


@pytest.mark.skipif(TOKEN_INVALID, reason='Token invalid')
def test_get_last_24h_data(test_data):
    data_24h = get_last_24_hours_data(test_data)
    assert data_24h['TimeType'].iloc[0] == 'past'
    assert len(data_24h) == 24


@pytest.mark.skipif(TOKEN_INVALID, reason='Token invalid')
def test_get_7_days_data(test_data):
    data_168h = get_last_7_days_data(test_data)
    assert data_168h['TimeType'].iloc[0] == 'past'
    assert len(data_168h) == 168


@pytest.mark.skipif(TOKEN_INVALID, reason='Token invalid')
def test_get_last_4_weeks_data(test_data):
    data_672h = get_last_4_weeks_data(test_data)
    assert data_672h['TimeType'].iloc[0] == 'past'
    assert len(data_672h) == 672


def test_adjust_time():
    # last_test_day = global_test_df.tail(10)
    # print(last_test_day)
    # future_test_day = adjust_time(last_test_day, n_days=7)
    # assert future_test_day['Time'].tail(1) == '2019-18-04 23:00:00'

    # TODO: need new montel token to test this or restore timestamps from strings in 'Time' column
    pass


def test_get_one_day_fill():
    # fill_24h = get_one_day_fill(global_test_df)
    # assert len(fill_24h) == 24
    # assert fill_24h['TimeType'] == 'fill'

    # TODO: need new montel token to test this or restore timestamps from strings in 'Time' column
    pass


def test_get_one_week_fill():
    # fill_168h = get_one_week_fill(global_test_df)
    # assert len(fill_168h) == 168
    # assert fill_168h['TimeType'] == 'fill'

    # TODO: need new montel token to test this or restore timestamps from strings in 'Time' column
    pass


def test_create_plot_df():
    # past = get_last_4_weeks_data(global_test_df)
    # prediction = adjust_time(global_test_df.iloc[-672].copy(), n_days=35)
    # prediction['TimeType'] = 'prediction'
    # plot_df = create_plot_df(past, prediction, 'one_week_prediction')
    # assert len(plot_df) == 841  # 672 + 168 + 1

    # TODO: need new montel token to test this or restore timestamps from strings in 'Time' column
    pass