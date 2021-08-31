import shutil
import pytest
import numpy as np
import os

from src.models.model_factory import ModelFactory
from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.data_splitter import DataSplitter, train_test_split


@pytest.fixture
def datasets():
    model_config = {
        'batch_size': 16,
        'window_size': 12
    }

    dataset = DatasetGenerator().get_dataset('2020-01-01', '2020-12-31', 'T23')
    train, test = train_test_split(dataset, 0.2)

    # Normalize the data
    dt = DataTransformer()
    train = dt.fit_transform(train)
    test = dt.transform_data(test)

    return train, test


@pytest.mark.parametrize('model_name', ['lstm', 'linear_regression', 'trivial',
                                        'nn'])
@pytest.mark.parametrize('gap', [0, 23, 167])
def test_model(model_name, datasets, gap):
    train, test = datasets
    model_config = {
        'batch_size': 16,
        'window_size': 12,
        'gap': gap,
        'num_features': 19,
        'num_layers': 1,
        'hidden_layer_size': 64,
        'epochs': 3,
        'p_param': 1,
        'd_param': 0,
        'q_param': 0,
        's_param': 24
    }
    # Create final training and testing datasets (batches and windows)
    splitter = DataSplitter(model_config)
    train_dataset = splitter.split(train)
    test_dataset = splitter.split(test)

    model = ModelFactory.get(model_name, model_config)
    model.train(train_dataset, test_dataset, model_config)
    prediction = model.predict(test_dataset)

    assert prediction.shape == (1756 - 12 - gap + 1,)

    shutil.rmtree(f'data/test_models/{model_name}/', ignore_errors=True)
    os.makedirs(f'data/test_models/{model_name}/')
    model.save(f'data/test_models/{model_name}/')

    new_model = ModelFactory.get(model_name, {})
    new_model.load(f'data/test_models/{model_name}/')
    new_prediction = new_model.predict(test_dataset)
    assert np.array_equal(prediction, new_prediction)
    shutil.rmtree(f'data/test_models/{model_name}/')


@pytest.mark.parametrize('model_name', ['linear'])
@pytest.mark.parametrize('gap', [0, 23, 167])
def test_model_linear(model_name, datasets, gap):
    train, test = datasets
    model_config = {
        'batch_size': 16,
        'window_size': 169,
        'gap': gap,
        'num_features': 19,
        'num_layers': 1,
        'hidden_layer_size': 64,
        'epochs': 3
    }
    # Create final training and testing datasets (batches and windows)
    splitter = DataSplitter(model_config)
    train_dataset = splitter.split(train)
    test_dataset = splitter.split(test)

    model = ModelFactory.get(model_name, model_config)
    model.train(train_dataset, test_dataset, model_config)
    prediction = model.predict(test_dataset)

    assert prediction.shape == (1756 - 169 - gap + 1,)

    shutil.rmtree(f'data/test_models/{model_name}/', ignore_errors=True)
    os.makedirs(f'data/test_models/{model_name}/')
    model.save(f'data/test_models/{model_name}/')

    new_model = ModelFactory.get(model_name, {})
    new_model.load(f'data/test_models/{model_name}/')
    new_prediction = new_model.predict(test_dataset)
    assert np.array_equal(prediction, new_prediction)
    shutil.rmtree(f'data/test_models/{model_name}/')


@pytest.mark.parametrize('model_name', ['sarimax'])
@pytest.mark.parametrize('gap', [0, 23, 167])
def test_model_sarimax(model_name, gap):

    model_config = {
        'batch_size': 16,
        'window_size': 12,
        'gap': gap,
        'num_features': 19,
        'num_layers': 1,
        'hidden_layer_size': 64,
        'epochs': 3
    }
    dataset = DatasetGenerator().get_dataset('2020-12-20', '2020-12-31', 'T23')

    # Normalize the data
    dt = DataTransformer()
    test = dt.fit_transform(dataset)
    # Create final training and testing datasets (batches and windows)
    splitter = DataSplitter(model_config)
    test_dataset = splitter.split(test)

    model = ModelFactory.get(model_name, model_config)
    model.train(None, test_dataset, model_config)
    prediction = model.predict(test_dataset)

    assert prediction.shape == (12*24 - 12 - gap,)

    shutil.rmtree(f'data/test_models/{model_name}/', ignore_errors=True)
    os.makedirs(f'data/test_models/{model_name}/')
    model.save(f'data/test_models/{model_name}/')

    new_model = ModelFactory.get(model_name, {})
    new_model.load(f'data/test_models/{model_name}/')
    new_prediction = new_model.predict(test_dataset)
    assert np.array_equal(prediction, new_prediction)
    shutil.rmtree(f'data/test_models/{model_name}/')

