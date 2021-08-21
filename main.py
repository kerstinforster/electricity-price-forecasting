# This is an example on how to use this project's code base.

import numpy as np

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.data_splitter import DataSplitter, train_test_split
from src.models.linear_regression_model import LinearRegressionModel
from src.models.lstm_model import LSTMModel

if __name__ == '__main__':
    model_params = {
        'batch_size': 64,
        'window_size': 7*24,
        'gap': 0,
        'num_features': 19,
        'num_layers': 1,
        'hidden_layer_size': 512
    }

    # Create a dataset generator class
    dg = DatasetGenerator(['all'])

    # Get a dataset
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')

    train, test = train_test_split(dataset, 0.2)

    # Create a data transformer for scaling data
    dt = DataTransformer()
    train = dt.fit_transform(train)
    test = dt.transform_data(test)

    # Split for training
    splitter = DataSplitter(model_params)
    train_dataset = splitter.split(train)
    test_dataset = splitter.split(test)

    # Get some testing sample:
    test_x = test.drop('Time', axis=1).values[:7*24, :]
    test_y = test.SPOTPrice.values[7*24]

    # Train linear regression model model
    #model = LinearRegressionModel({})
    #model.train(train_dataset, test_dataset, {})
    #prediction = model.predict(test_x)
    #print(f'True value: {test_y}')
    #print(f'Pred value: {prediction}')

    # Train LSTM model
    model = LSTMModel(model_params)
    model.train(train_dataset, test_dataset, {'epochs': 100})
    prediction = model.predict(test_x)
    print(f'True value: {test_y}')
    print(f'Pred value: {prediction}')

    # Test SARIMAX model
    # sarimax = SARIMAXModel({'gap': 0, 'spot_index': 0})
    # sarimax_data = train[30000:].append(test[:7*24]).drop(
    #     'Time', axis=1).T.values
    # print(f'SARIMAX Shape: {sarimax_data.shape}')
    # sarimax_pred = sarimax.predict(sarimax_data)
    # print(f'SARIMAX Pred value: {sarimax_pred}')

    # Revert the scaling of the prediction
    #print(f'Reverse-transformed prediction: \n'
    #      f'{dt.reverse_transform_spot(prediction)}')
