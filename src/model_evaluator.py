""" This file implements the ModelEvaluator that calculates up to 5 different
metrics to evaluate the performance of the time series predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelEvaluator(object):
    """
    This class
    """
    @staticmethod
    def _calc_r2(y_pred: np.array, y_true) -> float:
        """
        Calculates the r2 score / coefficient of determination
        :param y_pred: values predicted by the model
        :param y_true: actual values of the time-series data
        :return: r2 score / coefficient of determination
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def _calc_mae(y_pred: np.array, y_true: np.array) -> float:
        """
        Calculates the mean absolute error
        :param y_pred: values predicted by the model
        :param y_true: actual values of the time-series data
        :return: mean absolute error between the predicted y and the actual y
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _calc_mse(y_pred: np.array, y_true: np.array) -> float:
        """
        Calculates the mean squared error
        :param y_pred: values predicted by the model
        :param y_true: actual values of the time-series data
        :return: mean squared error between the predicted y and the actual y
        """
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def _calc_mape(y_pred: np.array, y_true: np.array) -> float:
        """
        Calculates the mean absolute percentage error
        :param y_pred: values predicted by the model
        :param y_true: actual values of the time-series data
        :return: mean absolute percentage error between the predicted y and the
        actual y
        """
        mask = (y_pred != 0)  # prevents divisions by 0
        return (np.fabs(y_pred - y_true) / y_pred)[mask].mean()

    @staticmethod
    def _calc_smape(y_pred: np.array, y_true: np.array) -> float:
        """
        Calculates the symmetric mean absolute percentage error
        :param y_pred: values predicted by the model
        :param y_true: actual values of the time-series data
        :return: symmetric mean absolute percentage error between the predicted
        y and the actual y
        """
        mask = (np.abs(y_true) + np.abs(y_pred) != 0)
        smape = 100 / len(y_true) * \
            np.sum(2 * np.abs(y_pred - y_true) /
                   (np.abs(y_true) + np.abs(y_pred))[mask])
        return smape

    @staticmethod
    def _create_comparison_plot(y_pred: pd.DataFrame, y_true: pd.DataFrame,
                                save_at: str, plot_name: str):
        """
        Creates a single plot with the data from the 2 given dataframes
        :param y_pred: dataframe containing 'Time' and 'PredictedSPOTPrice'
        :param y_true: dataframe containing 'Time' and 'SPOTPrice'
        :param save_at: path to the directory to save the plot images
        :param plot_name: name of the plot image. Has to end with .png
        :return: matplotlib fig and ax object of the plot
        """

        fig, ax = plt.subplots()
        y_pred.plot(kind='line', x='Time', y='PredictedSPOTPrice', ax=ax)
        y_true.plot(kind='line', x='Time', y='SPOTPrice', ax=ax)
        plt.grid()
        plt.legend()
        plt.xlabel('Time [h]')
        plt.ylabel('Price [€ / MWh]')
        plt.tight_layout()
        plt.savefig(save_at + plot_name)

        return fig, ax

    def create_all_comparison_plots(self, y_pred: pd.DataFrame,
                                    y_true: pd.DataFrame,
                                    save_at: str,
                                    n_steps: int = 72,
                                    show_plots: bool = False):
        """
        Creates 3 plots which visualize the our predictions compared to the
        labels. One plot shows the first n_steps of the test data, one plot
        shows the last n_steps (this is done so one can still see enough details
        in the plots). The 3rd plot shows all the test data and our predictions.

        :param y_pred: dataframe containing 'Time' and 'PredictedSPOTPrice'
        :param y_true: dataframe containing 'Time' and 'SPOTPrice'
        :param save_at: path to the directory to save the plot images
        :param n_steps: number of steps shown in the 2 smaller plots
        :param show_plots: flag whether to show the plots in the IDE
        :return:
        """

        start_data_pred = y_pred.head(n_steps)
        start_data_true = y_true.head(n_steps)

        end_data_pred = y_pred.tail(n_steps)
        end_data_true = y_true.tail(n_steps)

        fig_start, _ = self._create_comparison_plot(  # pylint: disable=unused-variable
            y_pred=start_data_pred,
            y_true=start_data_true,
            save_at=save_at,
            plot_name='comparison_start.png'
        )

        fig_end, _ = self._create_comparison_plot(  # pylint: disable=unused-variable
            y_pred=end_data_pred,
            y_true=end_data_true,
            save_at=save_at,
            plot_name='comparison_end.png'
        )

        fig_all, _ = self._create_comparison_plot(  # pylint: disable=unused-variable
            y_pred=y_pred,
            y_true=y_true,
            save_at=save_at,
            plot_name='comparison_all.png'
        )

        if show_plots:
            plt.show()

    def evaluate(self, y_pred: np.array, test_dataset: Any,
                 eval_metrics: list = None) -> dict:
        """
        Calculates all requested metrics for evaluation
        :param y_pred: values predicted by the model
        :param test_dataset: tf.data.Dataset with all test samples
            The dataset can be used as follows:
            for batch in dataset:
                x, y = batch
            Shapes: x -> (batch_size, window_size, 19(num_features));
                    y -> (batch_size,)
        :param eval_metrics: list of requested metrics (all, r2, mae, mse, mape,
        smape)
        :return: scores_dict with the score of each requested metric
        """
        y_pred = y_pred.reshape((-1,))
        total_target = np.empty(shape=(1))
        for batch in test_dataset:
            _, target = batch
            total_target = np.concatenate([total_target, target], axis=0)
        y_true = total_target.reshape((-1,))[-y_pred.size:]

        if eval_metrics is None:
            eval_metrics = ['all']

        scores_dict = {}
        if 'r2' in eval_metrics or 'all' in eval_metrics:
            scores_dict['r2_score'] = self._calc_r2(y_pred, y_true)

        if 'mae' in eval_metrics or 'all' in eval_metrics:
            scores_dict['mae_score'] = self._calc_mae(y_pred, y_true)

        if 'mse' in eval_metrics or 'all' in eval_metrics:
            scores_dict['mse_score'] = self._calc_mse(y_pred, y_true)

        if 'mape' in eval_metrics or 'all' in eval_metrics:
            scores_dict['mape_score'] = self._calc_mape(y_pred, y_true)

        if 'smape' in eval_metrics or 'all' in eval_metrics:
            scores_dict['smape_score'] = self._calc_smape(y_pred, y_true)

        if not scores_dict:
            raise ValueError('Valid values for eval_metrics are: all, r2, mae, '
                             'mse, mape or smape.')

        return scores_dict
