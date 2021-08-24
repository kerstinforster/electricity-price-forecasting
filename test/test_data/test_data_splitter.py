import pytest
import numpy as np

from src.data.data_splitter import DataSplitter
from src.data.dataset_generator import DatasetGenerator


@pytest.mark.parametrize("gap", [0, 23, 167])
def test_data_splitter(gap):
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2016-01-15', 'T23')
    # Dataset contains 15*24=360 entries
    # Window size is 168
    # This means we have 192-gap samples
    model_config = {
        'batch_size': 4,
        'window_size': 7 * 24,
        'gap': gap,
        'shuffle': False
    }
    splitter = DataSplitter(model_config)
    data = splitter.split(dataset)
    batch_num = 0
    data_index = 0
    for batch in data:
        batch_num += 1
        x, y = batch
        for element in range(x.shape[0]):
            sample = np.asarray(x[element, :, :])
            target = np.asarray(y[element])
            assert sample.shape == (168, 19)
            assert np.array_equal(sample,
                                  dataset.values[data_index:data_index+168, 1:])
            assert target.size == 1
            assert target == dataset.values[data_index + 168 + gap, 1]
            data_index += 1
    assert data_index == 192-gap

