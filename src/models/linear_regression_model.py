""" This class implements a linear regression model that predicts """

import numpy as np
from model_interface import BaseModel
from sklearn.linear_model import LinearRegression
from typing import Any
import pickle as pkl


class LinearRegressionModel(BaseModel):
    """
    This class implements linear regression as an example/baseline model
    for the eletricity price prediction task
    """

    def __init__(self, name: str = 'linear_regression'):
        """
        Constructor for the linear regression example/baseline model
        """
        super().__init__(name)
        self.model = LinearRegression(n_jobs=-1)
        self.model_trained = False

    def train(self, x_train: np.array, y_train: np.array, model_params: dict,
              save_at: str = None) -> Any:
        # TODO: check if this algo also takes multivariate regression tasks
        self.model.fit(x_train, y_train)
        self.model_trained = True

    def predict(self, x_input: np.array) -> np.array:
        if self.model_trained:
            return self.model.predict(x_input)
        else:
            raise ValueError("Linear Regression model has to be trained or "
                             "loaded before predictions can be made.")

    def save(self, path: str):
        if self.model_trained:
            # TODO: fix path to modelzoo here
            pkl.dump(self.model, path)
        else:
            raise ValueError("The linear regression model should be trained "
                             "before it is saved.")

    def load(self, path: str):
        self.model_trained = True
        # TODO: fix path to modelzoo here
        self.model = pkl.load(path)
