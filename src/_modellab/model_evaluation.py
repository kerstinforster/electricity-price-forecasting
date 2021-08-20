""" This file implements the """

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelEvaluator(object):
    """

    """

    @staticmethod
    def calc_smape(y_pred: np.array, y_true: np.array) -> float:
        smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) /
                                           (np.abs(y_true) + np.abs(y_pred)))
        return smape

    @staticmethod
    def calc_r2(y_pred: np.array, y_true) -> float:
        return r2_score(y_true, y_pred)

    @staticmethod
    def calc_mae(y_pred: np.array, y_true: np.array) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calc_mse(y_pred: np.array, y_true: np.array) -> float:
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def calc_mape(y_pred: np.array, y_true: np.array) -> float:
        mask = (y_pred != 0)  # prevents divisions through 0
        return (np.fabs(y_pred - y_true) / y_pred)[mask].mean()


# metrics are implemented now

# now we need a rolling window cross validation