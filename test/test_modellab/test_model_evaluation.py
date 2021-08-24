""" Validate that the model evaluation functions work as expected """

import pytest
import numpy as np
from src._modellab.model_evaluation import ModelEvaluator


def test_calc_r2():
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([2, 3, 4, 5, 6])

    r2_score = ModelEvaluator._calc_r2(y_pred, y_true)
    assert r2_score == 0.5


def test_calc_mae():
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([2, 3, 4, 5, 6])

    mae_score = ModelEvaluator._calc_mae(y_pred, y_true)
    assert mae_score == 1.0


def test_calc_mse():
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([2, 3, 4, 5, 6])

    mse_score = ModelEvaluator._calc_mse(y_pred, y_true)
    assert mse_score == 1.0


def test_calc_mape():
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([2, 3, 4, 5, 6])

    mape_score = ModelEvaluator._calc_mape(y_pred, y_true)
    assert np.isclose(mape_score, 0.45666666666666667)


def test_calc_smape():
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([2, 3, 4, 5, 6])

    smape_score = ModelEvaluator._calc_smape(y_pred, y_true)
    assert np.isclose(smape_score, 35.128427128427134)


def test_evaluate():
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([2, 3, 4, 5, 6])

    model_eval = ModelEvaluator()
    scores = model_eval.evaluate(y_pred, y_true, ['all'])
    scores = np.array(list(scores.values()))
    test_scores = np.array([0.5, 1.0, 1.0, 0.45666666666666667,
                            35.128427128427134])

    for i in range(scores.shape[0]):
        assert np.isclose(scores[i], test_scores[i])
