""" This class implements a linear model that predicts electricity prices """

import numpy as np
from typing import Any

from src.models.model_interface import BaseModel


class LinearModel(BaseModel):
    """
    This class implements a linear model
    for the electricity price prediction task
    """

    def __init__(self, model_params: dict, name: str = 'linear'):
        """
        Constructor for the linear model setting the name field and
        the specific model parameters
        :param name: name of the used algorithm, 'linear'
        :param model_params: Dictionary of parameters for the model
        """
        super().__init__(name, model_params)
        self.gap = model_params.get('gap', 0)

    def train(self, dataset: Any, test_dataset: Any, model_params: dict) -> Any:  # pylint: disable=unused-argument
        """
        Trains the linear model with the provided data
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
        prediction = np.empty(shape=(0, 1))
        for batch in test_dataset:
            x, _ = batch
            spot_weekago = np.asarray(x)[:, -169, 0].reshape((-1, 1))
            spot_weekagonext = np.asarray(x)[:, -169+self.gap+1, 0].reshape((-1, 1))
            spot_diff_weekago = spot_weekagonext - spot_weekago
            pred = np.asarray(x)[:, -1, 0].reshape((-1, 1)) + spot_diff_weekago
            prediction = np.concatenate([prediction, pred], axis=0)

        return prediction.reshape((-1,))
    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        For this model, no saving is necessary
        :param path: path and model name at location where model should be saved
        """
        pass

    def load(self, path: str):
        """
        Loads the model from the given path
        For this model, no loading is necessary
        :param path: path and model name at location where model should be
        loaded from
        """
        pass
