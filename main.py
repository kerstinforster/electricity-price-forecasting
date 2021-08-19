# This is an example on how to use this project's code base.

import pandas as pd

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.data_splitter import DataSplitter, train_test_split
from src.models.linear_regression_model import LinearRegressionModel
from src.models.sarimax_model import SARIMAXModel


if __name__ == '__main__':
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
    splitter = DataSplitter(7 * 24)  # One week training window size
    x_train, y_train = splitter.split(train, 0)  # Gap 0, 23, 167 here
    x_test, y_test = splitter.split(test, 0)
    print(f'X_Train: {x_train.shape}')
    print(f'Y_Train: {y_train.shape}')
    print(f'X_Test: {x_test.shape}')
    print(f'Y_Test: {y_test.shape}')

    # Train linear regression model model
    model = LinearRegressionModel({})
    model.train(x_train, y_train, {})
    prediction = model.predict(x_test[0, :, :])
    print(f'True value: {y_test[0, :]}')
    print(f'Pred value: {prediction}')

    # Test SARIMAX model
    # sarimax = SARIMAXModel({'gap': 0, 'spot_index': 0})
    # sarimax_data = train[30000:].append(test[:7*24]).drop(
    #     'Time', axis=1).T.values
    # print(f'SARIMAX Shape: {sarimax_data.shape}')
    # sarimax_pred = sarimax.predict(sarimax_data)
    # print(f'SARIMAX Pred value: {sarimax_pred}')

    # Revert the scaling of the prediction
    print(f'Reverse-transformed prediction: \n'
          f'{dt.reverse_transform_spot(prediction)}')
