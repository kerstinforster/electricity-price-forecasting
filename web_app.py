"""
This script creates an interactive web application in your browser at:
http://localhost:8501

To start the frontend type "streamlit run web_app.py" in your terminal in the
group03 repo
"""

from src.data.dataset_generator import DatasetGenerator
from src.frontend.frontend_utils import *
from src.frontend.frontend_graphs import GraphGenerator
from src.frontend.frontend_elements import *
from src.final_predictor import FinalPredictor


@st.cache
def request_montel_data() -> pd.DataFrame:
    """
    Fetches the electricity spot price data up to the current time.
    It unfortunately cannot be moved into 'src' because we cannot import from
    'data' into 'frontend' (application cannot resolve these paths and crashes).

    :return: Dataframe with columns 'Time' and 'SPOTPrice'
    """
    dg = DatasetGenerator(['montel'])
    date, time = get_date_and_time()
    dataset = dg.get_dataset('2021-08-01', date, time)
    return dataset


def get_date_and_time():
    from datetime import datetime
    date = datetime.now().strftime('%Y-%m-%d')
    hour = datetime.now().strftime('T%H')
    return date, hour


def request_prediction(
        df: pd.DataFrame,
        required_prediction: str) -> pd.DataFrame:
    """
    Requests a prediction from 'LivePredictionPipeline' for a specified time
    (1 hour, day or week).

    :param df: Dataframe with the necessary data for the prediction
    :param required_prediction: string that specifies time of the prediction
    :return: Dataframe with a single row 'Time', 'SPOTPrice', 'TimeType'
    """
    date, time = get_date_and_time()
    fp = FinalPredictor(date, time)

    if required_prediction == "one_hour_prediction":
        frame = df.iloc[-1].copy()
        frame['Time'] = frame['Time'] + pd.Timedelta(hours=1)
        frame['SPOTPrice'] = fp.predict_hour()
        frame['TimeType'] = 'prediction'
        return frame

    elif required_prediction == "one_day_prediction":
        frame = df.iloc[-1].copy()
        frame['Time'] = frame['Time'] + pd.Timedelta(hours=24)
        frame['SPOTPrice'] = fp.predict_day()
        frame['TimeType'] = 'prediction'
        return frame

    elif required_prediction == "one_week_prediction":
        frame = df.iloc[-1].copy()
        frame['Time'] = frame['Time'] + pd.Timedelta(hours=168)
        frame['SPOTPrice'] = fp.predict_week()
        frame['TimeType'] = 'prediction'
        return frame

    else:
        raise ValueError('required_prediction has an invalid value in the'
                         'prediction request')


graph_generator = GraphGenerator()
electricity_spot_data = request_montel_data()

# Putting together the frontend components:
create_header_logo()

st.title("Electricity price prediction:")

options, choice = create_options_dropdown()

# Options logic
if options[choice] == "one_hour_prediction":
    st.subheader("Electricity price in 1 hour:")

    past = get_last_24_hours_data(electricity_spot_data)
    prediction = request_prediction(past, 'one_hour_prediction')
    plot_df = create_plot_df(past, prediction, 'one_hour_prediction')

    graph_generator.create_final_prediction_plot(plot_df)
    create_comparison_table("24 hours ago", past, prediction)

elif options[choice] == "one_day_prediction":
    st.subheader("Electricity price in 1 day:")

    past = get_last_7_days_data(electricity_spot_data)
    prediction = request_prediction(past, 'one_day_prediction')
    plot_df = create_plot_df(past, prediction, 'one_day_prediction')

    graph_generator.create_final_prediction_plot(plot_df)
    create_comparison_table("1 week ago", past, prediction)

elif options[choice] == "one_week_prediction":
    st.subheader("Electricity price in 1 week:")

    past = get_last_4_weeks_data(electricity_spot_data)
    prediction = request_prediction(past, 'one_week_prediction')
    plot_df = create_plot_df(past, prediction, 'one_week_prediction')

    graph_generator.create_final_prediction_plot(plot_df)
    create_comparison_table("4 weeks ago", past, prediction)
