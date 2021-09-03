"""
This file is responsible for building all elements shown in the streamlit
frontend. Each element has to be called in web_app.py in the order one
wants to display them.
"""
import streamlit as st


def create_header_logo() -> None:
    """
    Creates the header logo at the top of the web application
    """
    st.image("src/frontend/resources/currence_logo_big.png")


def create_comparison_table(past_string, past, prediction) -> None:
    """
    Creates a table to compare the past, current and predicted electricity price
    :param past_string: prediction specific column label
    :param past: Dataframe with past data
    :param prediction: Dataframe with the prediction
    """
    st.subheader("Comparison")
    cols = st.columns(4)
    cols[1].write(past_string)  # column label
    cols[2].write("Current")  # column label
    cols[3].write("Predicted")  # column label

    cols = st.columns(4)
    cols[0].write("Price")  # row label
    cols[1].write(f'{past["SPOTPrice"].iloc[0]}')
    cols[2].write(f'{past["SPOTPrice"].iloc[-1]}')
    cols[3].write(f'{prediction["SPOTPrice"]}')

    cols = st.columns(4)
    cols[0].write("Time")  # row label
    cols[1].write(f'{past["Time"].iloc[0].time()}')
    cols[2].write(f'{past["Time"].iloc[-1].time()}')
    cols[3].write(f'{prediction["Time"].time()}')

    cols = st.columns(4)
    cols[0].write("Date")  # row label
    cols[1].write(f'{past["Time"].iloc[0].date()}')
    cols[2].write(f'{past["Time"].iloc[-1].date()}')
    cols[3].write(f'{prediction["Time"].date()}')


def create_options_dropdown() -> (dict, st.selectbox):
    """
    Creates a dropdown menu in which one can choose the desired prediction time.
    :return: dictionary with the options and the dropdown element
    """
    options = {
        "Predict electricity price in one hour": "one_hour_prediction",
        "Predict electricity price in one day": "one_day_prediction",
        "Predict electricity price in one week": "one_week_prediction"
        }

    choice = st.selectbox(
        "Which price do you want predicted?",
        tuple(options.keys()))

    return options, choice
