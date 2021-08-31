import os
import numpy as np

from src.models.lstm_model import LSTMModel
from src.model_evaluator import ModelEvaluator
from src.data.dataset_generator import DatasetGenerator
from src.data.data_transformer import DataTransformer
from src.data.data_splitter import DataSplitter, train_test_split


if __name__ == '__main__':
    folder = 'results'

    # Generate a dataset
    dg = DatasetGenerator(['all'])
    dataset = dg.get_dataset('2016-01-01', '2021-08-15', 'T23')

    # Split the dataset into train and test datasets
    train_raw, test_raw = train_test_split(dataset, 0.2)

    dt = DataTransformer()
    train = dt.fit_transform(train_raw)
    test = dt.transform_data(test_raw)

    scores = []
    folders = []

    for subfolder in os.listdir(folder):
        name = os.path.join(folder, subfolder)
        if not os.path.isdir(name): continue
        print(name)
        if name == 'results/lstm_1': continue

        model = LSTMModel({})
        model.load(name)

        model_config = {
            'batch_size': 64,
            'window_size': model.window_size,
            'gap': 0
        }
        splitter = DataSplitter(model_config)
        test_dataset = splitter.split(test)
        test_raw_dataset = splitter.split(test_raw)

        prediction = model.predict(test_dataset)

        score = ModelEvaluator().evaluate(dt.reverse_transform_spot(prediction),
                                           test_raw_dataset)

        scores.append(score)
        folders.append(name)

    indices = np.argsort([score['smape_score'] for score in scores])
    for ind in indices:
        print(scores[ind], folders[ind])



