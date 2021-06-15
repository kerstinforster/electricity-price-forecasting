""" Tests that validate the data getter methods """
import os
from typing import Any

import pytest

from src.data.data_getter import BaseDataGetter


class DataGetterTest(BaseDataGetter):
    def _get_raw_data(self) -> Any:
        pass

    def _process_raw_data(self, data: Any) -> Any:
        pass

    def __init__(self):
        super().__init__("test")


def test_init():
    dg = DataGetterTest()
    path = dg.get_project_root()
    assert os.path.exists(path)
    assert os.path.exists(
        os.path.join(path, 'test')
    )
    assert os.path.exists(
        os.path.join(path, 'README.md')
    )
    assert os.path.exists(
        os.path.join(path, 'data', 'test')
    )
    assert dg.data_dir == os.path.join(path, 'data', 'test')
    assert dg.name == "test"
    assert dg._root_dir == path

    # Test if path already exists
    dg = DataGetterTest()
    os.removedirs(os.path.join(path, 'data', 'test'))
