""" Final predictor used to perform the three predictions with the final models """

from src.models.model_factory import ModelFactory
from src.data.data_transformer import DataTransformer
from src.data.dataset_generator import DatasetGenerator


class FinalPredictor:
    def __init__(self, date: str, time: str):
        """
        Initialize a final predictor that predicts the three time horizons
        :param date: Date to start the prediction from
        :param time: Time to start the prediction from
        """
        self.date = date
        self.time = time

    def get_data(self, model_path: str, window_size: int) -> tuple:
        """
        Get data for the prediction step
        :param model_path: Model path to load the DataTransformer from
        :param window_size: Window Size that was used in the model
        :return: tuple: 1) np.array of shape (1, window_size, 19)
                        2) DataTransformer instance
        """
        dg = DatasetGenerator(['all'])
        dataset = dg.get_dataset('2021-08-01', self.date, self.time)
        transformer = DataTransformer.load(model_path)
        scaled_data = transformer.transform_data(dataset)
        data = scaled_data.drop('Time', axis=1)
        return (data.iloc[-window_size:, :].values.reshape((1, window_size, 19)),
                transformer)

    def predict_hour(self) -> float:
        """
        Predict one hour ahead SPOT Price
        :return: The predicted SPOT Price
        """
        data, transformer = self.get_data('results/gap_0', 336)
        model = ModelFactory.get('lstm', {})
        model.load('results/gap_0')
        prediction = model.predict(data)
        prediction = transformer.reverse_transform_spot(prediction)
        return prediction

    def predict_day(self) -> float:
        """
        Predict one day ahead SPOT Price
        :return: The predicted SPOT Price
        """
        data, transformer = self.get_data('results/gap_23', 168)
        model = ModelFactory.get('lstm', {})
        model.load('results/gap_0')
        prediction = model.predict(data)
        prediction = transformer.reverse_transform_spot(prediction)
        return prediction

    def predict_week(self) -> float:
        """
        Predict one week ahead SPOT Price
        :return: The predicted SPOT Price
        """
        data, transformer = self.get_data('results/gap_167', 168)
        model = ModelFactory.get('nn', {})
        model.load('results/gap_0')
        prediction = model.predict(data)
        prediction = transformer.reverse_transform_spot(prediction)
        return prediction
