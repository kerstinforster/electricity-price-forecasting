""" This class implements a linear regression model that predicts """

import numpy as np
import os
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
        Constructor for the linear regression model setting the name field and
        the specific model parameters
        :param name: name of the used algorithm, 'linear_regression'
        :param model_params: Dictionary of parameters for the model
        """
        super().__init__(name, model_params)
        self.model = LinearRegression(n_jobs=-1)
        self.window_size = model_params.get('window_size', 7 * 24)
        self.gap = model_params.get('gap', 0)
        self.n_features = model_params.get('num_features', 19)
        self.batch_size = model_params.get('batch_size', 64)

    def train(self, dataset: Any, test_dataset: Any, model_params: dict) -> Any:  # pylint: disable=unused-argument
        """
        Trains the linear regression model with the provided data
        :param dataset: Training dataset in format tf.data.Dataset
            The dataset can be used as follows:
            for batch in dataset:
                x, y = batch
            Shapes: x -> (batch_size, window_size, 19(num_features));
                    y -> (batch_size,)
        :param test_dataset: test dataset -> Can not be used for training,
            only for printing test loss or similar
        :param model_params: dictionary which sets the relevant hyperparameters
            for the training procedure
        """
        # For linear regression, the tf.data.Dataset is not directly usable.
        # Input for linear regression is (n_samples, n_features) dimensional.
        # This means we must extract all the data from the dataset into an array
        total_input = np.empty(shape=(0, self.window_size * self.n_features))
        total_target = np.empty(shape=(0,))
        for batch in dataset:
            x, target = batch
            x = np.reshape(x, (x.shape[0], -1))
            total_input = np.concatenate([total_input, x], axis=0)
            total_target = np.concatenate([total_target, target], axis=0)
        # print(f'Performing linear regression with input shape '
        #       f'{total_input.shape}')
        # print(f'\t and target shape: {total_target.shape}')
        self.model.fit(total_input, total_target)
        self.model_trained = True

    def predict(self, test_dataset: Any) -> np.array:
        """
        Uses the trained model to make a prediction based on x_input
        :param test_dataset: Training dataset in format tf.data.Dataset
            The dataset can be used as follows:
            for batch in dataset:
                x, y = batch
            Shapes: x -> (batch_size, window_size, 19(num_features));
                    y -> (batch_size,)
        :return: np.array containing all predictions, shape: (n_test,)
        """
        total_input = np.empty(shape=(0, self.window_size * self.n_features))
        for batch in test_dataset:
            x, _ = batch
            x = np.reshape(x, (x.shape[0], -1))
            total_input = np.concatenate([total_input, x], axis=0)
        if self.model_trained:
            return self.model.predict(total_input).reshape((-1,))
        else:
            raise ValueError('Linear Regression model has to be trained or '
                             'loaded before predictions can be made.')

    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: path and model name at location where model should be saved
        """
        if not self.model_trained:
            raise ValueError('The linear regression model should be trained '
                             'before it is saved.')
        with open(os.path.join(path, 'linear_regression.bin'), 'wb') as file:
            pkl.dump(self.model, file)
        with open(os.path.join(path, 'model_params.bin'), 'wb') as file:
            pkl.dump(self.model_params, file)


    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: path and model name at location where model should be
        loaded from
        """
        self.model_trained = True
        with open(os.path.join(path, 'linear_regression.bin'), 'rb') as file:
            self.model = pkl.load(file)
        with open(os.path.join(path, 'model_params.bin'), 'rb') as file:
            self.model_params = pkl.load(file)
        model_params = self.model_params
        self.window_size = model_params.get('window_size', 7 * 24)
        self.gap = model_params.get('gap', 0)
        self.n_features = model_params.get('num_features', 19)
        self.batch_size = model_params.get('batch_size', 64)
