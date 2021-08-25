"""
This file shows an example grid search for the lstm model
"""

from src.data.dataset_generator import DatasetGenerator
from src.data.data_splitter import train_test_split
from src.data.data_transformer import DataTransformer
from src.grid_searcher import GridSearcher


if __name__ == '__main__':
    # define a parameter grid for the SARIMAX model
    sarimax_param_grid = {
        'model_name': ['sarimax'],
        'window_size': [168],
        'gap': [0, 23, 167],  # Prediction horizons (hour, day, week)
        # 'param': [range(0, 3)]
    }

    trivial_param_grid = {
        'model_name': ['trivial']  # Add trivial model as baseline
    }

    # list of dicts containing the parameter grid for the every model
    all_model_param_grids = [sarimax_param_grid,
                             trivial_param_grid]

    # CREATE TRAIN AND TEST DATASET #
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')
    train, test = train_test_split(dataset, 0.1)
    dt = DataTransformer()
    train = dt.fit_transform(train)
    test = dt.transform_data(test)

    # Start the grid search
    grid_search = GridSearcher(all_model_param_grids)
    grid_search.run(train, test)

