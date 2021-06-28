import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px


from src.data.montel_data_getter import MontelDataGetter

# TODO: code refactoring
# TODO: code documentation

st.image("resources/currence_logo_big.png")

color_discrete_map = {
    'prediction': 'rgb(220, 80, 64)',
    'past': 'rgb(57, 57, 57)',
    'fill': 'rgb(190, 190, 190)'
    }


@st.cache
def request_montel_data():
    """

    :return:
    """
    dg = MontelDataGetter()
    return dg.get_data(end_date=get_current_time()[0])


def request_prediction(df, required_prediction):
    """

    :param df:
    :param required_prediction:
    :return:
    """
    if required_prediction == "one_hour_prediction":
        # TODO: this is a mock up, copies value 24h ago
        prediction = adjust_time(df.iloc[-24].copy(), n_days=1)
        prediction['time_type'] = 'prediction'
        return prediction

    elif required_prediction == "one_day_prediction":
        # TODO: this is a mock up, copies value 168h (1 week) ago
        prediction = adjust_time(df.iloc[-168].copy(), n_days=8)
        prediction['time_type'] = 'prediction'
        return prediction

    elif required_prediction == "one_week_prediction":
        # TODO: this is a mock up, copies value 672h (4 weeks) ago
        prediction = adjust_time(df.iloc[-672].copy(), n_days=35)
        prediction['time_type'] = 'prediction'
        return prediction

    else:
        raise ValueError('required_prediction has an invalid value in the'
                         'prediction request')


def get_current_time():
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M")
    return date, time


def get_last_24_hours_data(df):
    df = df.tail(24).copy()
    df['time_type'] = 'past'
    return df


def get_last_7_days_data(df):
    df = df.tail(168).copy()
    df['time_type'] = 'past'
    return df


def get_last_4_weeks_data(df):
    df = df.tail(672).copy()
    df['time_type'] = 'past'
    return df


def adjust_time(df, n_days):
    df['shifted_time'] = df.Time + pd.Timedelta(days=n_days)
    df['Time'] = df['shifted_time']
    df.drop(columns=['shifted_time'], inplace=True)
    return df


def get_one_day_fill(df):
    df = get_last_24_hours_data(df)
    # df = df.drop(df.tail(1).index)  # drop last element for prediction value
    df = adjust_time(df, n_days=1)
    df['time_type'] = 'fill'
    return df


def get_one_week_fill(df):
    df = get_last_7_days_data(df)
    df = adjust_time(df, n_days=7)
    df['time_type'] = 'fill'
    return df


def create_raw_data_plot(df):
    fig = px.bar(
        df, x='Time', y='Value',
        labels={
            "Time": "Time [h]",
            "Value": "Price [€ / MWh]"
        }
    )

    st.write(fig)


def create_one_hour_pred_plot(df):
    past = get_last_24_hours_data(df)
    prediction = request_prediction(past, 'one_hour_prediction')

    plot_df = past.append(prediction)
    # st.write(plot_df)

    fig = px.bar(plot_df, x='Time', y='Value',
                 color='time_type', color_discrete_map=color_discrete_map,
                 labels={
                     "Time": "Time [h]",
                     "Value": "Price [€ / MWh]"
                    }
                 )
    st.write(fig)

    return past, prediction


def create_one_day_pred_plot(df):
    past = get_last_7_days_data(df)
    future_fill = get_one_day_fill(df)
    prediction = request_prediction(past, 'one_day_prediction')

    plot_df = past.append(future_fill)
    plot_df = plot_df.append(prediction)
    plot_df.reset_index(inplace=True)

    # st.write(plot_df)

    fig = px.bar(plot_df, x='Time', y='Value',
                 color='time_type', color_discrete_map=color_discrete_map,
                 labels={
                     "Time": "Time [h]",
                     "Value": "Price [€ / MWh]"
                    }
                 )
    st.write(fig)

    return past, prediction


def create_one_week_pred_plot(df):
    past = get_last_4_weeks_data(df)
    future_fill = get_one_week_fill(df)
    prediction = request_prediction(past, 'one_week_prediction')

    plot_df = past.append(future_fill)
    plot_df = plot_df.append(prediction)
    plot_df.reset_index(inplace=True)

    # st.write(plot_df)

    fig = px.bar(plot_df, x='Time', y='Value',
                 color='time_type', color_discrete_map=color_discrete_map,
                 labels={
                     "Time": "Time [h]",
                     "Value": "Price [€ / MWh]"
                    }
                 )
    st.write(fig)

    return past, prediction


# loading all available spot data in a dataframe
spot_data = request_montel_data()

# Inspect raw data
# st.header("Inspect raw data")
# st.write(spot_data)
# create_raw_data_plot(spot_data)

# st.write(get_current_time())
# st.write(spot_data["Time"].tail(1))


st.title("Electricity price prediction:")


options = {
    "Predict electricity price in one hour": "one_hour_prediction",
    "Predict electricity price in one day": "one_day_prediction",
    "Predict electricity price in one week": "one_week_prediction"
}

choice = st.selectbox(
    "Which price do you want to predicted?",
    tuple(options.keys()))


def build_table(past_string, past, prediction):
    st.subheader("Comparison")
    cols = st.beta_columns(4)
    cols[1].write(past_string)
    cols[2].write("Current")
    cols[3].write("Predicted")

    cols = st.beta_columns(4)
    cols[0].write("Price")
    cols[1].write(f'{past["Value"].iloc[0]}')
    cols[2].write(f'{past["Value"].iloc[-1]}')
    cols[3].write(f'{prediction["Value"]}')

    cols = st.beta_columns(4)
    cols[0].write("Time")
    cols[1].write(f'{past["Time"].iloc[0]}')
    cols[2].write(f'{past["Time"].iloc[-1]}')
    cols[3].write(f'{prediction["Time"]}')


if options[choice] == "one_hour_prediction":
    st.subheader("Electricity price in 1 hour:")
    past, prediction = create_one_hour_pred_plot(spot_data)
    build_table("24 hours ago", past, prediction)

elif options[choice] == "one_day_prediction":
    st.subheader("Electricity price in 1 day:")
    past, prediction = create_one_day_pred_plot(spot_data)
    build_table("1 week ago", past, prediction)

elif options[choice] == "one_week_prediction":
    st.subheader("Electricity price in 1 week:")
    past, prediction = create_one_week_pred_plot(spot_data)
    build_table("4 week ago", past, prediction)

