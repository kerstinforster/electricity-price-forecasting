""" Test the data transformer """

import pytest

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.montel_data_getter import MontelDataGetter


def test_transforming():
    dataset = DatasetGenerator().get_dataset('2018-01-01', '2019-01-01', 'T16')
    transformer = DataTransformer()
    transformed_data = transformer.fit_transform(dataset)
    assert transformed_data.equals(transformer.transform_data(dataset))
    reverse_data = transformer.reverse_transform(transformed_data)

    assert dataset.round(4).equals(reverse_data.round(4))


def test_reverse_spot():
    dataset = DatasetGenerator().get_dataset('2018-01-01', '2019-01-01', 'T16')
    transformer = DataTransformer()
    transformed_data = transformer.fit_transform(dataset)
    reverse_spot_data = transformer.reverse_transform(
        transformed_data).SPOTPrice.values
    spot_data = transformer.reverse_transform_spot(transformed_data.SPOTPrice)
    for i in range(spot_data.size):
        assert spot_data[i] == reverse_spot_data[i]
        assert spot_data[i].round(4) == dataset.SPOTPrice.values[i].round(4)