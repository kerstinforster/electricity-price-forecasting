# This is an example on how to use this project's code base.

from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer


if __name__ == '__main__':
    # Create a dataset generator class
    dg = DatasetGenerator(['all'])

    # Get a dataset
    dataset = dg.get_dataset('2016-01-01', '2021-06-24', 'T16')
    print(f'Dataset: \n{dataset}')
    # Get a dataset with the latest data
    dataset_latest = dg.get_dataset('2016-01-01', 'latest', 'ignored')
    print(f'Latest data: \n{dataset_latest}')

    # Create a data transformer for scaling data
    dt = DataTransformer()
    # Scale the data
    scaled_data = dt.transform_data(dataset_latest)
    print(f'Scaled data: \n{scaled_data}')

    prediction = scaled_data.tail(5)  # Only for testing, no real prediction
    # Revert the scaling of the prediction
    print(f'Reverse-transformed prediction: \n'
          f'{dt.reverse_transform(prediction)}')
