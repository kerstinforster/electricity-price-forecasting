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

    #import numpy as np
    #a = dataset.to_numpy()
    #b = reverse_data.round(4).to_numpy()
    #x, y = np.where(a!=b)
    #for x_, y_ in zip(x,y):
    #    print(f"x{x_}/y{y_}/true{a[x_,y_]}/round{b[x_,y_]}")

    assert dataset.round(4).equals(reverse_data.round(4))