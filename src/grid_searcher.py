""" Grid searcher class which implements rudimentary grid search
functionality"""

from alive_progress import alive_bar
import itertools
from typing import Any
import os
import json

from src.model_evaluator import ModelEvaluator
from src.data.data_splitter import DataSplitter
from src.models.model_factory import ModelFactory
from src.data.data_transformer import DataTransformer


class GridSearcher:
    """
    Grid Searcher class that implements an experiment grid search
    """

    def __init__(self, parameter_grids: list):
        """
        Initialize the grid search class
        :param parameter_grids: List of dictionaries, each dictionary
            configuring the grid search for one model
        """
        self.parameter_grids = parameter_grids
        self.results = []
        self.evaluator = ModelEvaluator()
        if not os.path.exists('results'):
            os.makedirs('results')
        self.results_file_path = 'results/gs_results.json'
        if os.path.exists(self.results_file_path):
            raise RuntimeError(
                f'There is already a grid search results file '
                f'located in {self.results_file_path} ! In order '
                f'to not overwrite these results, the grid '
                f'search is stopped. Please rename the old file!')

    def run(self, train_dataset: Any, test_dataset: Any):
        for model_grid in self.parameter_grids:
            configs = self.get_all_combinations(model_grid)
            with alive_bar(len(configs),
                           title=f'Model {model_grid["model_name"]}',
                           force_tty=1, theme='smooth') as bar:
                for model_config in configs:
                    scores = self.train_model_config(
                        model_config, train_dataset, test_dataset)
                    self.results.append((model_config, scores))
                    bar()  # pylint: disable=not-callable
        with open(self.results_file_path, 'w', encoding='utf-8') as file:
            json.dump(self.results, file)

    @staticmethod
    def get_all_combinations(model_grid: dict):
        return list(dict(zip(model_grid.keys(), values)) for values in
                    itertools.product(*model_grid.values()))

    def train_model_config(self, model_config: dict, train_dataset: Any,
                           test_dataset: Any):
        dt = DataTransformer()
        train_norm = dt.fit_transform(train_dataset)
        test_norm = dt.transform_data(test_dataset)

        splitter = DataSplitter(model_config)
        train = splitter.split(train_norm)
        test = splitter.split(test_norm)
        test_raw = splitter.split(test_dataset)

        model = ModelFactory.get(model_config['model_name'], model_config)
        model.train(train, test, model_config)

        model_path = self.get_unique_model_path(model_config)
        os.makedirs(model_path)
        model.save(model_path)
        dt.save(model_path)

        prediction = model.predict(test)
        prediction = dt.reverse_transform_spot(prediction)
        return self.evaluator.evaluate(prediction, test_raw)

    @staticmethod
    def get_unique_model_path(model_config: dict) -> str:
        config = model_config.copy()
        path = model_config['model_name']
        config.pop('model_name')
        for key, value in config.items():
            path += f'_{key}-{str(value)}'
        return os.path.join('results', path)
