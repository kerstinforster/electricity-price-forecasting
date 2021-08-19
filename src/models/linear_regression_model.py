""" This class implements a linear regression model that predicts """

import numpy as np
import pickle as pkl
from typing import Any
from sklearn.linear_model import LinearRegression

from src.models.model_interface import BaseModel


class LinearRegressionModel(BaseModel):
    """
    This class implements linear regression as an example/baseline model
    for the electricity price prediction task
    """

    def __init__(self, model_params: dict, name: str = 'linear_regression'):
        """
        Constructor for the linear regression example/baseline model
        """
        super().__init__(name, model_params)
        self.model = LinearRegression(n_jobs=-1)

    def train(self, x_train: np.array, y_train: np.array, model_params: dict,  # pylint: disable=unused-argument
              save_at: str = None) -> Any:                                     # pylint: disable=unused-argument
        """
        Trains a model with the provided data (x_train, y_train)
        :param x_train: np.array with (n_samples X n_features X window_length)
        :param y_train: np.array with (n_samples X n_labels)
        :param model_params: dictionary which sets the relevant hyperparameters
        :param save_at: saves model at given path with given name if not None
        :return: trained instance of the model
        """
        # Input for linear regression is (n_samples, n_features) dimensional
        x = x_train.reshape(x_train.shape[0],
                            (x_train.shape[1] * x_train.shape[2]))

        self.model.fit(x, y_train)
        self.model_trained = True

    def predict(self, x_input: np.array) -> np.array:
        """
        Uses the trained model to make a prediction based on x_input
        :param x_input: single timestep with n_features X n_step_size
        :return: Correctly timestamped pandas dataframe with the predicted
        value/s
        """
        x = x_input.reshape(1, x_input.shape[0] * x_input.shape[1])
        if self.model_trained:
            return self.model.predict(x)
        else:
            raise ValueError('Linear Regression model has to be trained or '
                             'loaded before predictions can be made.')

    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: path and model name at location where model should be saved
        """
        if self.model_trained:
            # TODO: fix path to modelzoo here
            pkl.dump(self.model, path)
        else:
            raise ValueError('The linear regression model should be trained '
                             'before it is saved.')

    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: path and model name at location where model should be
        loaded from
        """
        self.model_trained = True
        # TODO: fix path to modelzoo here
        self.model = pkl.load(path)
