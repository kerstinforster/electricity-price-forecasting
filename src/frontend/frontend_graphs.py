""" This class implements the plotting functionality for the frontend """
import plotly.express as px
import streamlit as st


class GraphGenerator(object):
    """
    This class is responsible for creating and displaying all necessary graph /
    plot elements in the frontend prototype.
    """

    def __init__(self):
        self.color_discrete_map = {
            'prediction': 'rgb(24, 37, 85)',
            'past': 'rgb(100, 100, 100)',
            'fill': 'rgb(190, 190, 190)'
        }

        self.time_price_labels = {
            'Time': 'Time [h]',
            'Value': 'Price [€ / MWh]'
        }

    def create_raw_data_plot(self, plot_df) -> None:
        """
        Visualizes any given dataframe with the columns 'Time' and 'SPOTPrice'
        as an interactive bar plot
        :param plot_df: dataframe to be plotted
        """
        fig = px.bar(
            plot_df, x='Time', y='SPOTPrice',
            labels=self.time_price_labels
        )
        st.write(fig)

    def create_final_prediction_plot(self, plot_df) -> None:
        """
        Visualizes any given dataframe with the columns 'Time', 'SPOTPrice' and
        'TimeType' as an interactive bar plot
        :param plot_df: dataframe to be plotted
        """
        fig = px.bar(plot_df, x='Time', y='SPOTPrice',
                     color='TimeType',
                     color_discrete_map=self.color_discrete_map,
                     labels=self.time_price_labels
                     )
        st.write(fig)
