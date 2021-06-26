# This is an example on hopw to use this project's code base.

from src.data.dataset_generator import DatasetGenerator


if __name__ == '__main__':
    # Get the dataset
    dg = DatasetGenerator(['all'])

    dataset = dg.get_dataset('2016-01-01', '2021-06-24', 'T16')
    print(dataset)
    dataset_latest = dg.get_dataset('2016-01-01', 'latest', 'ignored')
    print(dataset_latest)
