""" This class implements a linear regression model that predicts """

import numpy as np
from typing import Any
import tensorflow as tf
import os
import pickle as pkl

from src.models.model_interface import BaseModel


class LSTMModel(BaseModel):
    """
    This class implements an LSTM network as model
    for the electricity price prediction task
    """

    def __init__(self, model_params: dict, name: str = 'lstm'):
        """
        Constructor for the lstm model setting the name field and the
        specific model parameters
        :param name: name of the used algorithm, 'lstm'
        :param model_params: Dictionary of parameters for the model
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

    def train(self, dataset: Any, test_dataset: Any, model_params: dict) -> Any:  # pylint: disable=unused-argument
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
        epochs = model_params.get('epochs', 100)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, mode='min',
            restore_best_weights=True)

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        self.history = self.model.fit(dataset, epochs=epochs,
                                      validation_data=test_dataset,
                                      callbacks=[early_stopping])

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
        prediction = self.model.predict(test_dataset)
        return prediction.reshape((-1,))

    def save(self, path: str):
        """
        Saves the model at the given path with the given name
        :param path: path and model name at location where model should be saved
        """
        self.model.save(path)
        with open(os.path.join(path, 'model_params.bin'), 'wb') as file:
            pkl.dump(self.model_params, file)

    def load(self, path: str):
        """
        Loads the model from the given path
        :param path: path and model name at location where model should be
        loaded from
        """
        self.model = tf.keras.models.load_model(path)
        with open(os.path.join(path, 'model_params.bin'), 'rb') as file:
            self.model_params = pkl.load(file)
        model_params = self.model_params
        self.hidden_layer_size = model_params.get('hidden_layer_size', 128)
        self.num_layers = model_params.get('num_layers', 1)
        self.input_size = model_params.get('num_features', 19)
        self.batch_size = model_params.get('batch_size', 64)
        self.window_size = model_params.get('window_size', 7 * 24)
