"""
This file shows an example grid search for the linear models
"""

from src.data.dataset_generator import DatasetGenerator
from src.data.data_splitter import train_test_split
from src.data.data_transformer import DataTransformer
from src.grid_searcher import GridSearcher


if __name__ == '__main__':
    # define a parameter grid for each model
    linear_param_grid = {
        'model_name': ['linear'],
        'window_size': [169, 2000, 3000, 4500, 5500],
        'gap': [0, 23, 167],  # Prediction horizons (hour, day, week)

    }

    #linear_regression_param_grid = {
    #   'model_name': ['linear_regression'],
    #    'window_size': [12, 24, 72, 168],
    #    'gap': [0, 23, 167],
    #}

    trivial_param_grid = {
        'model_name': ['trivial'],  # Add trivial model as baseline
        'gap': [0, 23, 167]
    }

    # list of dicts containing the parameter grid for the every model
    all_model_param_grids = [linear_param_grid,
                             #linear_regression_param_grid,
                             trivial_param_grid]

    # CREATE TRAIN AND TEST DATASET #
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')

    train, test = train_test_split(dataset, 0.5)
    # Start the grid search
    grid_search = GridSearcher(all_model_param_grids)
    grid_search.run(train, test)

