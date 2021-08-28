""" This file implements the SARIMAX model."""

from typing import Any
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.model_interface import BaseModel


class SARIMAXModel(BaseModel):
    """
    Implementation of the SARIMAX model for prediction
    """

    def __init__(self, model_params: dict, name='sarimax'):
        """
        Constructor for the sarimax model setting the name field and
        the specific model parameters
        :param name: name of the used algorithm, 'sarimax'
        :param model_params: Dictionary of parameters for the model
        """
        super().__init__(name, model_params)
        # assert 'gap' in model_params.keys()
        # if model_params['gap'] == 0:
        #     self.p_param = (1, 0, 0)    # Values from grid search
        #     self.s_param = (1, 0, 0, 24)
        # else:
        #     self.p_param = (2, 0, 1)
        #     self.s_param = (2, 0, 1, 24)
        self.gap = model_params.get('gap', 0)
        self.param = (model_params.get('p_param', 1),
                      model_params.get('d_param', 0),
                      model_params.get('q_param', 0))
        self.s_param = (model_params.get('p_param', 1),
                        model_params.get('d_param', 0),
                        model_params.get('q_param', 0),
                        model_params.get('s_param', 24))
        self.model = None
        self.fit_model = None
        self.training_data = None
        self.training_exog = None

    def train(self, dataset: Any, test_dataset: Any, model_params: dict) -> Any:
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
        pass

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
        prediction = []
        for batch in test_dataset:
            inputs, _ = batch
            data = inputs[:, :, 0].numpy()
            for slices in data:
                self.model = SARIMAX(slices, order=self.param,
                                     seasonal_order=self.s_param)
                self.fit_model = self.model.fit(method='powell', disp=False)
                forecast = self.fit_model.predict(len(slices) + self.gap,
                                                  len(slices) + self.gap)
                prediction = np.concatenate((prediction, forecast))

        return prediction

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
