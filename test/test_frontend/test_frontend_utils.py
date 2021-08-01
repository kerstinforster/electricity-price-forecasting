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
