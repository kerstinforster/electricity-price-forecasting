"""
This file is responsible for building all elements shown in the streamlit
frontend. Each element has to be called in web_app.py in the order one
wants to display them.
"""
from datetime import datetime
import pandas as pd


def get_current_time() -> (str, str):
    """
    Gets the current date and time
    :return: tuple of strings with date (yyyy-mm-dd) and time (HH:MM)
    """
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M')
    return date, time


def get_last_24_hours_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the last 24h of the time-series data and labels them as 'past'
    :param df: Dataframe that contains all the time-series data
    :return: Dataframe with 24 rows
    """
    df = df.tail(24).copy()
    df['TimeType'] = 'past'
    return df


def get_last_7_days_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the last 168h of the time-series data and labels them as 'past'
    :param df: Dataframe that contains all the time-series data
    :return: Dataframe with 168 rows
    """
    df = df.tail(168).copy()
    df['TimeType'] = 'past'
    return df


def get_last_4_weeks_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the last 672h of the time-series data and labels them as 'past'
    :param df: Dataframe that contains all the time-series data
    :return: Dataframe with 672 rows
    """
    df = df.tail(672).copy()
    df['TimeType'] = 'past'
    return df


def adjust_time(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    Shifts the time-stamps of a dataframe n_days into the future. Used for
    adjusting the time of the filler data and the prediction for the final plot.
    :param df: Dataframe that should be adjusted
    :param n_days: number of days the time-stamps are shifted into the future
    :return: Dataframe with adjusted time-stamps
    """
    df['shifted_time'] = df.Time + pd.Timedelta(days=n_days)
    df['Time'] = df['shifted_time']
    df.drop(columns=['shifted_time'], inplace=True)
    return df


def get_one_day_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 24h of filler data for the final plot by copying the last 24h and
    labels them as 'fill'. Used for the one_day_prediction.
    :param df: Dataframe that contains all the time-series data
    :return: Dataframe with the "next" 24h of data
    """
    df = get_last_24_hours_data(df)
    df = adjust_time(df, n_days=1)
    df['TimeType'] = 'fill'
    return df


def get_one_week_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 168h of filler data for the final plot by copying the last 168h and
    labels them as 'fill'. Used for the one_week_prediction.
    :param df: Dataframe that contains all the time-series data
    :return: Dataframe with the "next" 168h of data
    """
    df = get_last_7_days_data(df)
    df = adjust_time(df, n_days=7)
    df['TimeType'] = 'fill'
    return df


def create_plot_df(past: pd.DataFrame,
                   prediction: pd.DataFrame,
                   required_prediction: str) -> pd.DataFrame:
    """
    Creates the dataframe necessary for the final prediction plot in the
    frontend.
    :param past: past data up to the current time
    :param prediction: predicted electricity price
    :param required_prediction: given timeframe of the prediction
    :return: Dataframe with past, filler and prediction data for visualization
    """

    if required_prediction == 'one_hour_prediction':
        plot_df = past.append(prediction)

    elif required_prediction == 'one_day_prediction':
        filler = get_one_day_fill(past)
        past = past.tail(72)  # reducing past to 3 days for better visibility
        plot_df = past.append(filler)
        plot_df = plot_df.append(prediction)

    elif required_prediction == 'one_week_prediction':
        filler = get_one_week_fill(past)
        past = past.tail(168)  # reducing past to 1 week for better visibility
        plot_df = past.append(filler)
        plot_df = plot_df.append(prediction)

    else:
        raise ValueError('required_prediction has an invalid value in '
                         'create_plot_df')

    plot_df.reset_index(inplace=True)
    return plot_df
