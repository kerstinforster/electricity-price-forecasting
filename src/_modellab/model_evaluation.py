""" This file implements the """

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelEvaluator(object):
    """

    """
    @staticmethod
    def _calc_r2(y_pred: np.array, y_true) -> float:
        return r2_score(y_true, y_pred)

    @staticmethod
    def _calc_mae(y_pred: np.array, y_true: np.array) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _calc_mse(y_pred: np.array, y_true: np.array) -> float:
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def _calc_mape(y_pred: np.array, y_true: np.array) -> float:
        mask = (y_pred != 0)  # prevents divisions through 0
        return (np.fabs(y_pred - y_true) / y_pred)[mask].mean()

    @staticmethod
    def _calc_smape(y_pred: np.array, y_true: np.array) -> float:
        smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) /
                                           (np.abs(y_true) + np.abs(y_pred)))
        return smape

    def evaluate(self, y_pred: np.array, y_true: np.array,
                 eval_metrics=None) -> dict:

        if eval_metrics is None:
            eval_metrics = ['all']

        scores_dict = dict()
        if 'r2' in eval_metrics or 'all' in eval_metrics:
            scores_dict["r2_score"] = self._calc_r2(y_pred, y_true)

        elif 'mae' in eval_metrics or 'all' in eval_metrics:
            scores_dict["mae_score"] = self._calc_mae(y_pred, y_true)

        elif 'mse' in eval_metrics or 'all' in eval_metrics:
            scores_dict["mse_score"] = self._calc_mse(y_pred, y_true)

        elif 'mape' in eval_metrics or 'all' in eval_metrics:
            scores_dict["mape_score"] = self._calc_mape(y_pred, y_true)

        elif 'smape' in eval_metrics or 'all' in eval_metrics:
            scores_dict["smape_score"] = self._calc_smape(y_pred, y_true)

        else:
            raise ValueError("Valid values for eval_metrics are: all, r2, mae, "
                             "mse, mape or smape.")

        return scores_dict
