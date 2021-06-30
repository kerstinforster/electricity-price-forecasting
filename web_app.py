"""
This script creates an interactive web application in your browser at:
http://localhost:8501

To start the frontend type "streamlit run web_app.py" in your terminal in the
group03 repo
"""

import streamlit as st
import pandas as pd
from src.data.montel_data_getter import MontelDataGetter
from src.frontend.frontend_utils import *
from src.frontend.frontend_graphs import GraphGenerator
from src.frontend.frontend_elements import *


@st.cache
def request_montel_data() -> pd.DataFrame:
    """
    Fetches the electricity spot price data up to the current time.
    It unfortunately cannot be moved into 'src' because we cannot import from
    'data' into 'frontend' (application cannot resolve these paths and crashes).

    :return: Dataframe with columns 'Time' and 'Value'
    """
    dg = MontelDataGetter()
    current_date = get_current_time()[0]
    return dg.get_data(end_date=current_date)


def request_prediction(
        df: pd.DataFrame,
        required_prediction: str) -> pd.DataFrame:
    """
    Currently still a mock-up function! This function can also not be moved into
    'src', because it will have to use 'LivePredictionPipeline'.

    Requests a prediction from 'LivePredictionPipeline' for a specified time
    (1 hour, day or week).

    :param df: Dataframe with the necessary data for the prediction
    :param required_prediction: string that specifies time of the prediction
    :return: Dataframe with a single row 'Time', 'Value', 'TimeType'
    """
    if required_prediction == "one_hour_prediction":
        # TODO: this is a mock up, copies value 24h ago
        prediction = adjust_time(df.iloc[-24].copy(), n_days=1)
        prediction['TimeType'] = 'prediction'
        return prediction

    elif required_prediction == "one_day_prediction":
        # TODO: this is a mock up, copies value 168h (1 week) ago
        prediction = adjust_time(df.iloc[-168].copy(), n_days=8)
        prediction['TimeType'] = 'prediction'
        return prediction

    elif required_prediction == "one_week_prediction":
        # TODO: this is a mock up, copies value 672h (4 weeks) ago
        prediction = adjust_time(df.iloc[-672].copy(), n_days=35)
        prediction['TimeType'] = 'prediction'
        return prediction

    else:
        raise ValueError('required_prediction has an invalid value in the'
                         'prediction request')


graph_generator = GraphGenerator()
electricity_spot_data = request_montel_data()
# Putting together the frontend components:

create_header_logo()

st.title("Electricity price prediction:")

options, choice = create_options_dropdown()

# options logic
if options[choice] == "one_hour_prediction":
    st.subheader("Electricity price in 1 hour:")
    plot_df = create_plot_df()
    past, prediction = create_one_hour_pred_plot(electricity_spot_data)
    create_comparison_table("24 hours ago", past, prediction)

elif options[choice] == "one_day_prediction":
    st.subheader("Electricity price in 1 day:")
    past, prediction = create_one_day_pred_plot(electricity_spot_data)
    create_comparison_table("1 week ago", past, prediction)

elif options[choice] == "one_week_prediction":
    st.subheader("Electricity price in 1 week:")
    past, prediction = create_one_week_pred_plot(electricity_spot_data)
    create_comparison_table("4 week ago", past, prediction)
