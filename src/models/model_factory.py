""" This class implements a factory for models"""

from src.models.linear_regression_model import LinearRegressionModel
from src.models.lstm_model import LSTMModel
from src.models.trivial_model import TrivialModel
from src.models.nn_model import NeuralNetworkModel


class ModelFactory:
    """
    Factory for model instances
    """

    @staticmethod
    def get(model_name: str, model_params: dict):
        """
        Get a model instance based on the name of the model/algorithm
        :param model_name: name of the model/algorithm
        :param model_params: dict with all the hyper-parameters for the
        requested model (only one each)
        :return: class instance of the model
        """

        if model_name == 'linear_regression':
            return LinearRegressionModel(model_params)

        elif model_name == 'lstm':
            return LSTMModel(model_params)

        elif model_name == 'trivial':
            return TrivialModel(model_params)

        elif model_name == 'nn':
            return NeuralNetworkModel(model_params)

        else:
            raise ValueError(f'The model with the name "{model_name}" '
                             f'does not exist!')
