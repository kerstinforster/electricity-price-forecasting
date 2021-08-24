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
        :param model_params: Dictionary of parameters for the model
        """
        self.name = name
        self.model_params = model_params
        self.model = None
        self.model_trained = False

    @abstractmethod
    def train(self, dataset: Any, test_dataset: Any, model_params: dict) \
            -> None:
        """
        Trains the model with the provided data
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
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
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
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: Path to store the model at
        """
        raise NotImplementedError('This is an abstract class method.')

    @abstractmethod
    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: Path to load the model from
        """
        raise NotImplementedError('This is an abstract class method.')
