""" Tests that validate the data getter methods """
import os

import pytest

from src.data.data_getter import *


class DataGetterTest(DataGetter):
    def __init__(self):
        super().__init__("test")

    def get_data(self) -> None:
        pass


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


def test_get_data_uses_derived():
    dg = DataGetterTest()
    # Base class would throw error
    dg.get_data()
    path = dg.get_project_root()
    os.removedirs(os.path.join(path, 'data', 'test'))
    with pytest.raises(TypeError):
        _ = DataGetter("base")
