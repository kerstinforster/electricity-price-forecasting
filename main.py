""" This is an example script that shows how to use this project's code base.
It shows our data collection and processing, and how to train some of
our prediction models. """

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.data_splitter import DataSplitter, train_test_split
from src.models.linear_regression_model import LinearRegressionModel
from src.models.lstm_model import LSTMModel
from src.models.trivial_model import TrivialModel
from src.model_evaluator import ModelEvaluator


if __name__ == '__main__':
    model_config = {
        'batch_size': 64,
        'window_size': 7*24,
        'gap': 0,
        'num_features': 19,
        'num_layers': 1,
        'hidden_layer_size': 512,
    }

    # Generate a dataset
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')

    # Split the dataset into train and test datasets
    train_raw, test_raw = train_test_split(dataset, 0.2)

    # Normalize the data
    dt = DataTransformer()
    train = dt.fit_transform(train_raw)
    test = dt.transform_data(test_raw)

    # Create final training and testing datasets (batches and windows)
    splitter = DataSplitter(model_config)
    train_dataset = splitter.split(train)
    test_dataset = splitter.split(test)
    test_raw_dataset = splitter.split(test_raw)

    model = LinearModel(model_config)
    model.train(train_dataset, test_dataset, {})
    trivial_prediction = model.predict(test_dataset)
    model_evaluator = ModelEvaluator()
    print(f'Linear Model Scores: \n '
          f'{model_evaluator.evaluate(trivial_prediction, test_dataset)}')

    model = TrivialModel(model_config)
    model.train(train_dataset, test_dataset, {})
    trivial_prediction = model.predict(test_dataset)
    model_evaluator = ModelEvaluator()
    trivial_prediction = dt.reverse_transform_spot(trivial_prediction)
    print(f'Trivial Model Scores: \n '
          f'{model_evaluator.evaluate(trivial_prediction, test_raw_dataset)}')

    # Train linear regression model
    model = LinearRegressionModel(model_config)
    model.train(train_dataset, test_dataset, {})
    linear_prediction = model.predict(test_dataset)
    linear_prediction = dt.reverse_transform_spot(linear_prediction)
    model_evaluator = ModelEvaluator()
    print(f'Linear Regression Model Scores: \n '
          f'{model_evaluator.evaluate(linear_prediction, test_raw_dataset)}')

    # Train LSTM model
    model = LSTMModel(model_config)
    model.train(train_dataset, test_dataset, {'epochs': 10})
    lstm_prediction = model.predict(test_dataset)
    lstm_prediction = dt.reverse_transform_spot(lstm_prediction)
    model_evaluator = ModelEvaluator()
    print(f'LSTM Model Scores: \n '
          f'{model_evaluator.evaluate(lstm_prediction, test_raw_dataset)}')
