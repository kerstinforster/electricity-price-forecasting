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
        self.window_size = model_params.get('window_size', 7 * 24)
        self.gap = model_params.get('gap', 0)
        self.n_features = model_params.get('num_features', 19)
        self.batch_size = model_params.get('batch_size', 64)

    def train(self, dataset: Any, test_dataset: Any, model_params: dict,  # pylint: disable=unused-argument
              save_at: str = None) -> Any:                                # pylint: disable=unused-argument
        """
        Trains a model with the provided data (x_train, y_train)
        :param dataset: np.array with (n_samples X n_features X window_length)
        :param test_dataset: np.array with (n_samples X n_labels)
        :param model_params: dictionary which sets the relevant hyperparameters
        :param save_at: saves model at given path with given name if not None
        :return: trained instance of the model
        """
        # Input for linear regression is (n_samples, n_features) dimensional
        total_input = np.empty(shape=(1, self.window_size * self.n_features))
        total_target = np.empty(shape=(1))
        for batch in dataset:
            input, target = batch
            input = np.reshape(input, (input.shape[0], -1))
            total_input = np.concatenate([total_input, input], axis=0)
            total_target = np.concatenate([total_target, target], axis=0)
        print(f'Performing linear regression with input shape {total_input.shape}')
        print(f'\t and target shape: {total_target.shape}')
        self.model.fit(total_input, total_target)
        self.model_trained = True

    def predict(self, x_input: np.array) -> np.array:
        """
        Uses the trained model to make a prediction based on x_input
        :param x_input: single timestep with n_step_size X n_features
        :return: Correctly timestamped pandas dataframe with the predicted
        value/s
        """
        x = np.reshape(x_input, (1, -1))

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
