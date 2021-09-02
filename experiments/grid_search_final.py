"""
This file performs the final grid search hoping to get the best model randomly
"""

from src.data.dataset_generator import DatasetGenerator
from src.data.data_splitter import train_test_split
from src.data.data_transformer import DataTransformer
from src.grid_searcher import GridSearcher


if __name__ == '__main__':
    # define a parameter grid for each model
    gap0_param_grid = {
        'model_name': ['lstm'],
        'i': list(range(10)),  # Train 10 times and then choose best
        'window_size': [336],
        'gap': [0],
        'num_layers': [1],
        'hidden_layer_size': [256],
        'epochs': [100],
        'regularization': [0],
        'want': [10.3]
    }

    gap23_param_grid = {
        'model_name': ['lstm'],
        'i': list(range(10)),  # Train 10 times and then choose best
        'window_size': [168],
        'gap': [23],
        'num_layers': [2],
        'hidden_layer_size': [256],
        'epochs': [100],
        'regularization': [0],
        'want': [25.3]
    }

    gap167_param_grid = {
        'model_name': ['nn'],
        'i': list(range(10)),
        'gap': [167],
        'window_size': [168],
        'first_HL_size': [64],
        'second_HL_size': [64],
        'drop': [0.3],
        'activ': ['relu'],
        'want': [27.3]
    }

    # list of dicts containing the parameter grid for the every model
    all_model_param_grids = [gap0_param_grid,
                             gap23_param_grid,
                             gap167_param_grid]

    # CREATE TRAIN AND TEST DATASET #
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')
    train, test = train_test_split(dataset, 0.1)

    # Start the grid search
    grid_search = GridSearcher(all_model_param_grids)
    grid_search.run(train, test)

