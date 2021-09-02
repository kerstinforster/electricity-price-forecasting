""" Test the data transformer """

import shutil
import os
import pandas as pd

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer


def test_transforming():
    dataset = DatasetGenerator().get_dataset('2018-08-01', '2019-01-01', 'T16')
    transformer = DataTransformer()
    transformed_data = transformer.fit_transform(dataset)
    assert transformed_data.equals(transformer.transform_data(dataset))
    reverse_data = transformer.reverse_transform(transformed_data)

    assert dataset.round(4).equals(reverse_data.round(4))


def test_reverse_spot():
    dataset = DatasetGenerator().get_dataset('2018-08-01', '2019-01-01', 'T16')
    transformer = DataTransformer()
    transformed_data = transformer.fit_transform(dataset)
    transformed_data2 = transformer.transform_data(dataset)
    pd.testing.assert_frame_equal(transformed_data, transformed_data2)
    reverse_spot_data = transformer.reverse_transform(
        transformed_data).SPOTPrice.values
    spot_data = transformer.reverse_transform_spot(transformed_data.SPOTPrice.values)
    for i in range(spot_data.size):
        assert spot_data[i] == reverse_spot_data[i]
        assert spot_data[i].round(4) == dataset.SPOTPrice.values[i].round(4)


def test_save_load():
    dataset = DatasetGenerator().get_dataset('2018-08-01', '2019-01-01', 'T16')
    transformer = DataTransformer()
    transformed_data = transformer.fit_transform(dataset)
    shutil.rmtree(f'data/test_models/transformer/', ignore_errors=True)
    os.makedirs(f'data/test_models/transformer/')
    transformer.save('data/test_models/transformer/')

    new_transformer = DataTransformer.load('data/test_models/transformer/')
    transformed_data2 = new_transformer.transform_data(dataset)
    pd.testing.assert_frame_equal(transformed_data, transformed_data2)
    shutil.rmtree(f'data/test_models/transformer/', ignore_errors=True)
