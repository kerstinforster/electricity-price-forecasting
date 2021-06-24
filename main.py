# This is an example on hopw to use this project's code base.

from src.data.dataset_generator import DatasetGenerator


if __name__ == '__main__':
    # Get the dataset
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2010-01-01', '2020-12-31')

    print(dataset.head())

