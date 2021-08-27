""" This class implements a linear model that predicts electricity prices """

import numpy as np
from typing import Any
import os
import pickle as pkl
from src.models.model_interface import BaseModel
from sklearn.linear_model import LinearRegression

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
        self.model = LinearRegression(n_jobs=-1)
        self.gap = model_params.get('gap', 0)
        self.batch_size = model_params.get('batch_size', 64)

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
        prediction = np.empty(shape=(0, 1))
        for batch in test_dataset:
            x, _ = batch
            preds = np.empty(shape=(0,1))
            for i in np.arange(x.shape[0]):
                spot_weekago = np.asarray(x)[i, -169::-168, 0]
                spot_weekago_next = np.asarray(x)[i, -169 + self.gap+1:self.gap:-168, 0]
                spot_diff_weekago = spot_weekago_next - spot_weekago
                input = np.arange(spot_diff_weekago.shape[0]).reshape(-1, 1)
                self.model.fit(input, spot_diff_weekago)
                pred = np.asarray(x)[i, -1, 0] + self.model.predict([input[-1]+1])
                pred = pred.reshape(-1, 1)
                preds = np.concatenate([preds, pred], axis=0)
            prediction = np.concatenate([prediction,preds],axis=0)
        return prediction.reshape((-1,))

    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        For this model, no saving is necessary
        :param path: path and model name at location where model should be saved
        """
        with open(os.path.join(path, 'linear.bin'), 'wb') as file:
            pkl.dump(self.model, file)
        with open(os.path.join(path, 'model_params.bin'), 'wb') as file:
            pkl.dump(self.model_params, file)

    def load(self, path: str):
        """
        Loads the model from the given path
        For this model, no loading is necessary
        :param path: path and model name at location where model should be
        loaded from
        """
        self.model_trained = True
        with open(os.path.join(path, 'linear.bin'), 'rb') as file:
            self.model = pkl.load(file)
        with open(os.path.join(path, 'model_params.bin'), 'rb') as file:
            self.model_params = pkl.load(file)
        model_params = self.model_params
        self.window_size = model_params.get('window_size', 7 * 24+1)
        self.gap = model_params.get('gap', 0)
        self.n_features = model_params.get('num_features', 19)
        self.batch_size = model_params.get('batch_size', 64)
