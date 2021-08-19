""" This file contains the base class for the prediction models"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseModel(ABC):
    """
    An abstract interface for all prediction models
    """
    def __init__(self, name: str, model_params: dict):
        """
        Constructor for the prediction model setting the name field and the
        specific model parameters
        :param name: name of the used algorithm, for example 'linear_regression'
        """
        self.name = name
        self.model_params = model_params
        self.model = None
        self.model_trained = False

    @abstractmethod
    def train(self, x_train: np.array, y_train: np.array, model_params: dict,
              save_at: str = None) -> Any:
        """
        Trains a model with the provided data (x_train, y_train)
        :param x_train: np.array with (n_timesteps X n_features X n_step_size)
        :param y_train: np.array with (n_timesteps X n_labels)
        :param model_params: dictionary which sets the relevant hyperparameters
        :param save_at: saves model at given path with given name if not None
        :return: trained instance of the model
        """
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
    def predict(self, x_input: np.array) -> np.array:
        """
        Uses the trained model to make a prediction based on x_input
        :param x_input: single timestep with n_features X n_step_size
        :return: Correctly timestamped pandas dataframe with the predicted
        value/s
        """
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: path and model name in modelzoo
        """
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: path and model name in modelzoo
        """
        raise NotImplementedError('This is an abstract class method.')
