""" Test the data transformer """

import pytest

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.montel_data_getter import MontelDataGetter

try:
    _ = MontelDataGetter()
    TOKEN_INVALID = False
except ConnectionRefusedError:
    TOKEN_INVALID = True


@pytest.mark.skipif(TOKEN_INVALID, reason='Token invalid')
def test_transforming():
    dataset = DatasetGenerator().get_dataset('2016-01-01', '2017-01-01', 'T23')
    transformer = DataTransformer()
    transformed_data = transformer.transform_data(dataset)
    reverse_data = transformer.reverse_transform(transformed_data)
    assert dataset.equals(reverse_data.round(4))
