"""
This file shows an example grid search for the nn model
"""

from src.data.dataset_generator import DatasetGenerator
from src.data.data_splitter import train_test_split
from src.data.data_transformer import DataTransformer
from src.grid_searcher import GridSearcher
from keras.activations import relu, elu


if __name__ == '__main__':
    # define a parameter grid for each model
    nn_param_grid = {
        'model_name': ['nn'],
        'window_size': [12, 24, 36, 48, 96, 168, 336],
        'gap': [0, 23, 167],  # Prediction horizons (hour, day, week)
        'first_HL_size': [8, 64, 128, 256],
        'second_HL_size': [8, 64, 128, 256],
        #'batch_size': [32, 64],
        #'epochs': [25, 50, 100],
        'drop': [0, 0.1, 0.5],
        'activ': ['relu', 'elu']
    }

    trivial_param_grid = {
        'model_name': ['trivial']  # Add trivial model as baseline
    }

    # list of dicts containing the parameter grid for the every model
    all_model_param_grids = [nn_param_grid,
                             trivial_param_grid]

    # CREATE TRAIN AND TEST DATASET #
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')
    train, test = train_test_split(dataset, 0.1)

    # Start the grid search
    grid_search = GridSearcher(all_model_param_grids)
    grid_search.run(train, test)

