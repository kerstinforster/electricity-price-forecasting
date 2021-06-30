import plotly.express as px
import streamlit as st
from frontend_utils import *


class GraphGenerator(object):
    """
    This class is responsible for creating and displaying all necessary graph /
    plot elements in the frontend prototype.
    """

    def __init__(self):
        self.color_discrete_map = {
            'prediction': 'rgb(220, 80, 64)',
            'past': 'rgb(57, 57, 57)',
            'fill': 'rgb(190, 190, 190)'
        }

        self.time_price_labels = {
            "Time": "Time [h]",
            "Value": "Price [€ / MWh]"
        }

    def create_raw_data_plot(self, df):
        """

        :param df:
        :return:
        """
        fig = px.bar(
            df, x='Time', y='Value',
            labels=self.time_price_labels
        )
        st.write(fig)

    def create_one_hour_pred_plot(self, df):
        past = get_last_24_hours_data(df)
        prediction = request_prediction(past, 'one_hour_prediction')

        plot_df = past.append(prediction)

        fig = px.bar(plot_df, x='Time', y='Value',
                     color='TimeType',
                     color_discrete_map=self.color_discrete_map,
                     labels=self.time_price_labels
                     )
        st.write(fig)

        return past, prediction

    def create_one_day_pred_plot(self, df):
        past = get_last_7_days_data(df)
        future_fill = get_one_day_fill(df)
        prediction = request_prediction(past, 'one_day_prediction')

        plot_df = past.append(future_fill)
        plot_df = plot_df.append(prediction)
        plot_df.reset_index(inplace=True)

        fig = px.bar(plot_df, x='Time', y='Value',
                     color='TimeType',
                     color_discrete_map=self.color_discrete_map,
                     labels=self.time_price_labels
                     )

        st.write(fig)
        return past, prediction

    def create_one_week_pred_plot(self, df):
        past = get_last_4_weeks_data(df)
        future_fill = get_one_week_fill(df)
        prediction = request_prediction(past, 'one_week_prediction')

        plot_df = past.append(future_fill)
        plot_df = plot_df.append(prediction)
        plot_df.reset_index(inplace=True)

        # st.write(plot_df)

        fig = px.bar(plot_df, x='Time', y='Value',
                     color='TimeType',
                     color_discrete_map=self.color_discrete_map,
                     labels=self.time_price_labels
                     )

        st.write(fig)
        return past, prediction