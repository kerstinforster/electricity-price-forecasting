""" This class implements a linear regression model that predicts """

import numpy as np
from typing import Any
import tensorflow as tf

from src.models.model_interface import BaseModel


class LSTMModel(BaseModel):
    """
    This class implements an LSTM network as model
    for the electricity price prediction task
    """

    def __init__(self, model_params: dict, name: str = 'lstm'):
        """
        Constructor for the linear regression example/baseline model
        """
        super().__init__(name, model_params)
        self.hidden_layer_size = model_params.get('hidden_layer_size', 128)
        self.num_layers = model_params.get('num_layers', 1)
        self.input_size = model_params.get('num_features', 19)
        self.batch_size = model_params.get('batch_size', 64)
        self.window_size = model_params.get('window_size', 7*24)
        self.model = tf.keras.models.Sequential()
        layer_size = self.hidden_layer_size if isinstance(
            self.hidden_layer_size, int) else self.hidden_layer_size[0]
        self.model.add(tf.keras.layers.LSTM(
            layer_size,
            input_shape=(self.window_size, self.input_size),
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)))
        for i in range(self.num_layers - 1):
            layer_size = self.hidden_layer_size if isinstance(
                self.hidden_layer_size, int) else self.hidden_layer_size[i]
            self.model.add(tf.keras.layers.LSTM(
                layer_size, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
            ))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=1))

        self.history = None

    def train(self, x_train, y_train, model_params: dict,  # pylint: disable=unused-argument
              save_at: str = None) -> Any:                 # pylint: disable=unused-argument
        """
        Trains a model with the provided data (x_train, y_train)
        :param x_train: np.array with (n_samples X n_features X window_length)
        :param y_train: np.array with (n_samples X n_labels)
        :param model_params: dictionary which sets the relevant hyperparameters
        :param save_at: saves model at given path with given name if not None
        :return: trained instance of the model
        """
        epochs = model_params.get('epochs', 100)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, mode='min',
            restore_best_weights=True)

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        self.history = self.model.fit(x_train, epochs=epochs,
                                      validation_data=y_train,
                                      callbacks=[early_stopping])

    def predict(self, x_input: np.array) -> np.array:
        """
        Uses the trained model to make a prediction based on x_input
        :param x_input: single timestep with n_step_size X n_features
        :return: Correctly timestamped pandas dataframe with the predicted
        value/s
        """
        x_input = np.expand_dims(x_input, 0)
        prediction = self.model.predict(x_input)
        return prediction

    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: path and model name at location where model should be saved
        """
        pass

    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: path and model name at location where model should be
        loaded from
        """
        pass
