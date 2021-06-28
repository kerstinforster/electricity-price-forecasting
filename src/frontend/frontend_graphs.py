""" This class implements the plotting functionality for the frontend """
import plotly.express as px
import streamlit as st


class GraphGenerator(object):
    """
<<<<<<< HEAD
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
            "Time": "Time [h]",
            "Value": "Price [€ / MWh]"
        }

    def create_raw_data_plot(self, plot_df) -> None:
        """
        Visualizes any given dataframe with the columns 'Time' and 'Value' as an
        interactive bar plot
        :param plot_df: dataframe to be plotted
        """
        fig = px.bar(
            plot_df, x='Time', y='Value',
            labels=self.time_price_labels
        )
        st.write(fig)

    def create_final_prediction_plot(self, plot_df) -> None:
        """
        Visualizes any given dataframe with the columns 'Time', 'Value' and
        'TimeType' as an interactive bar plot
        :param plot_df: dataframe to be plotted
        """
        fig = px.bar(plot_df, x='Time', y='Value',
                     color='TimeType',
                     color_discrete_map=self.color_discrete_map,
                     labels=self.time_price_labels
                     )
        st.write(fig)
=======

    """
>>>>>>> feat: streamlit frontend prototype, logo, options, graph, table
