""" This file implements the SARIMAX model."""

from typing import Any
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.model_interface import BaseModel


class SARIMAXModel(BaseModel):
    """
    Implementation of the SARIMAX model for prediction
    """

    def __init__(self, model_params: dict):
        """
        Constructor for the SARIMAX model setting the name field and the
        specific model parameters
        :param name: name of the used algorithm
        """
        super().__init__('sarimax', model_params)
        assert 'gap' in model_params.keys()
        if model_params['gap'] == 0:
            self.p_param = (0, 2, 0)    # Values from grid search
            self.s_param = (0, 2, 0, 24)
        else:
            self.p_param = (2, 1, 1)
            self.s_param = (2, 1, 1, 24)
        self.model = None
        self.fit_model = None
        self.training_data = None
        self.training_exog = None

    def train(self, x_train: np.array, y_train: np.array, model_params: dict,
              save_at: str = None) -> Any:
        """
        Trains a model with the provided data (x_train, y_train)
        :param x_train: np.array with (n_timesteps X n_features)
        :param y_train: np.array with (n_timesteps X n_labels)
        :param model_params: dictionary which sets the relevant hyperparameters
        :param save_at: saves model at given path with given name if not None
        :return: nothing
        """
        pass

    def predict(self, x_input: np.array) -> np.array:
        """
        Make a prediction using the SARIMAX model
        :param x_input: single timestep with n_features X n_time_steps
        :return: Correctly timestamped pandas dataframe with the predicted
        value/s
        """
        spot_index = self.model_params.get('spot_index', 0)
        self.model = SARIMAX(x_input[spot_index, :], order=self.p_param,
                             seasonal_order=self.s_param)
        self.fit_model = self.model.fit()
        prediction = self.fit_model.predict(
            x_input.shape[1] + self.model_params['gap'])
        return prediction

    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: path and model name in modelzoo
        """
        pass

    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: path and model name in modelzoo
        """
        pass
