""" This is an example script that shows how to use this project's code base.
It shows our data collection and processing, and how to train some of
our prediction models. """

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.data_splitter import DataSplitter, train_test_split
from src.models.linear_regression_model import LinearRegressionModel
from src.models.lstm_model import LSTMModel
from src.models.nn_model import NeuralNetworkModel
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
    train, test = train_test_split(dataset, 0.2)

    # Normalize the data
    dt = DataTransformer()
    train = dt.fit_transform(train)
    test = dt.transform_data(test)

    # Create final training and testing datasets (batches and windows)
    splitter = DataSplitter(model_config)
    train_dataset = splitter.split(train)
    test_dataset = splitter.split(test)

    model = TrivialModel(model_config)
    model.train(train_dataset, test_dataset, {})
    trivial_prediction = model.predict(test_dataset)
    model_evaluator = ModelEvaluator()
    print(f'Trivial Model Scores: \n '
          f'{model_evaluator.evaluate(trivial_prediction, test_dataset)}')

    # Train Neural Network model
    model = NeuralNetworkModel(model_config)
    model.train(train_dataset, test_dataset, {'epochs': 100})
    nn_prediction = model.predict(test_dataset)
    model_evaluator = ModelEvaluator()
    print(f'Neural Network Model Scores: \n '
          f'{model_evaluator.evaluate(nn_prediction, test_dataset)}')

    # Train linear regression model
    model = LinearRegressionModel(model_config)
    model.train(train_dataset, test_dataset, {})
    linear_prediction = model.predict(test_dataset)
    model_evaluator = ModelEvaluator()
    print(f'Linear Regression Model Scores: \n '
          f'{model_evaluator.evaluate(linear_prediction, test_dataset)}')

    # Train LSTM model
    model = LSTMModel(model_config)
    model.train(train_dataset, test_dataset, {'epochs': 10})
    lstm_prediction = model.predict(test_dataset)
    model_evaluator = ModelEvaluator()
    print(f'LSTM Model Scores: \n '
          f'{model_evaluator.evaluate(lstm_prediction, test_dataset)}')


    # Revert the scaling of the prediction (to show how it works)
    print(f'Reverse-transformed Linear prediction: \n'
          f'{dt.reverse_transform_spot(linear_prediction[0])}')
    print(f'Reverse-transformed LSTM prediction: \n'
          f'{dt.reverse_transform_spot(lstm_prediction[0])}')
    print(f'Reverse-transformed Neural Network prediction: \n'
          f'{dt.reverse_transform_spot(nn_prediction[0])}')
