import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "models", "aussie_rain.joblib"))


weather_aus = load_model()


@st.cache_data
def load_and_analyze_data():
    raw_data = pd.read_csv(os.path.join(BASE_DIR, "data", "weatherAUS.csv"))

    rain_stats = (
        raw_data.groupby("Location")["RainTomorrow"]
        .value_counts(normalize=True)
        .unstack()
        .reset_index()
    )
    rain_stats["RainProbability"] = (rain_stats["Yes"] * 100).round(1)
    top_rain_cities = (
        rain_stats.sort_values(by="RainProbability", ascending=False)
        .head(5)[["Location", "RainProbability"]]
        .reset_index(drop=True)
    )
    top_rain_cities.index += 1
    num_cols_list = weather_aus["numeric_cols"]
    default_values = raw_data[num_cols_list].mean().round(1).to_dict()

    location = sorted(raw_data["Location"].unique())
    all_wind_dirs = sorted(raw_data["WindGustDir"].dropna().unique())

    default_values["Cloud9am"] = int(raw_data["Cloud9am"].median())
    default_values["Cloud3pm"] = int(raw_data["Cloud3pm"].median())
    wind_dir_9am = raw_data["WindDir9am"].mode()[0]
    wind_dir_3pm = raw_data["WindDir3pm"].mode()[0]
    return (
        raw_data,
        top_rain_cities,
        default_values,
        location,
        all_wind_dirs,
        wind_dir_9am,
        wind_dir_3pm,
    )


(
    raw_data,
    top_rain_cities,
    default_values,
    location,
    all_wind_dirs,
    wind_dir_9am,
    wind_dir_3pm,
) = load_and_analyze_data()

st.title("🌧️ Australian Rain Prediction Predictor")
st.markdown(
    "This app uses the **Random Forest** model to predict whether it will rain in Australia tomorrow, based on meteorological data."
)
st.image(os.path.join(BASE_DIR, "images", "australia.jpg"))

st.header("Statistics on observations in Australia")
st.subheader("Top 5 Cities with Highest Rain Probability")
st.dataframe(top_rain_cities)

st.bar_chart(top_rain_cities, x="Location", y="RainProbability")

st.header("Input Features")

col1, col2 = st.columns(2)

with col1:
    cities = st.selectbox("Select a City", location)
    min_temp = st.number_input(
        "Minimum Temperature (°C)", value=float(default_values["MinTemp"])
    )
    max_temp = st.number_input(
        "Maximum Temperature (°C)", value=float(default_values["MaxTemp"])
    )
    rainfall = st.number_input("Rainfall (mm)", value=float(default_values["Rainfall"]))
    rain_today = st.selectbox("Rain Today", ["No", "Yes"])

with col2:
    humidity_9am = st.slider(
        "Humidity at 9 AM (%)",
        min_value=0,
        max_value=100,
        value=int(default_values["Humidity9am"]),
    )
    humidity_3pm = st.slider(
        "Humidity at 3 PM (%)",
        min_value=0,
        max_value=100,
        value=int(default_values["Humidity3pm"]),
    )
    wind_gust_dir = st.selectbox("Wind Gust Direction", all_wind_dirs)
    wind_gust_speed = st.number_input(
        "Wind Gust Speed (km/h)", value=float(default_values["WindGustSpeed"])
    )
    wind_speed_9am = st.number_input(
        "Wind Speed at 9 AM (km/h)", value=float(default_values["WindSpeed9am"])
    )

with st.expander("Advanced Weather Settings (Optional)"):
    st.markdown(
        "The values shown here are the averages for this climate zone, but you can change them:"
    )
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        evaporation = st.number_input(
            "Evaporation (mm)", value=float(default_values["Evaporation"])
        )
        wind_speed_3pm = st.number_input(
            "Wind Speed at 3 PM (km/h)", value=float(default_values["WindSpeed3pm"])
        )
        pressure_3pm = st.number_input(
            "Pressure at 3 PM (hPa)",
            step=0.1,
            value=float(default_values["Pressure3pm"]),
        )
        pressure_9am = st.number_input(
            "Pressure at 9 AM (hPa)",
            step=0.1,
            value=float(default_values["Pressure9am"]),
        )
        wind_dir_9am_index = all_wind_dirs.index(wind_dir_9am)
        wind_dir_9am_widget = st.selectbox(
            "Wind Direction at 9 AM", all_wind_dirs, index=wind_dir_9am_index
        )
        wind_dir_3pm_index = all_wind_dirs.index(wind_dir_3pm)
        wind_dir_3pm_widget = st.selectbox(
            "Wind Direction at 3 PM", all_wind_dirs, index=wind_dir_3pm_index
        )

    with exp_col2:
        sunshine = st.number_input(
            "Sunshine (hours)", value=float(default_values["Sunshine"])
        )

        cloud_9am = st.slider(
            "Cloud Cover at 9 AM",
            min_value=0,
            max_value=8,
            value=int(default_values["Cloud9am"]),
        )
        cloud_3pm = st.slider(
            "Cloud Cover at 3 PM",
            min_value=0,
            max_value=8,
            value=int(default_values["Cloud3pm"]),
        )
        temp_9am = st.number_input(
            "Temperature at 9 AM (°C)", value=float(default_values["Temp9am"])
        )
        temp_3pm = st.number_input(
            "Temperature at 3 PM (°C)", value=float(default_values["Temp3pm"])
        )
st.markdown("### A comparison of humidity and temperature at 9 am and 3 pm")

stat_data = raw_data[raw_data["Location"] == cities][["Temp9am", "Temp3pm"]].mean()
stat_data.index = ["9 AM", "3 PM"]
st.bar_chart(stat_data)

if st.button("Predict Rain Tomorrow"):
    input_data = pd.DataFrame(
        {
            "MinTemp": [min_temp],
            "MaxTemp": [max_temp],
            "Rainfall": [rainfall],
            "WindGustSpeed": [wind_gust_speed],
            "WindSpeed9am": [wind_speed_9am],
            "Humidity3pm": [humidity_3pm],
            "Humidity9am": [humidity_9am],
            "Location": [cities],
            "WindGustDir": [wind_gust_dir],
            "RainToday": [rain_today],
            "WindDir9am": [wind_dir_9am_widget],
            "WindDir3pm": [wind_dir_3pm_widget],
            "Evaporation": [evaporation],
            "Sunshine": [sunshine],
            "WindSpeed3pm": [wind_speed_3pm],
            "Pressure9am": [pressure_9am],
            "Pressure3pm": [pressure_3pm],
            "Cloud9am": [cloud_9am],
            "Cloud3pm": [cloud_3pm],
            "Temp9am": [temp_9am],
            "Temp3pm": [temp_3pm],
        }
    )

    numeric_cols = input_data[weather_aus["numeric_cols"]]
    categorical_cols = input_data[weather_aus["categorical_cols"]]
    X_num = weather_aus["scaler"].transform(
        weather_aus["imputer"].transform(numeric_cols)
    )
    X_cat = weather_aus["encoder"].transform(categorical_cols)
    input_data = np.hstack((X_num, X_cat))

    prediction = weather_aus["model"].predict(input_data)
    prediction_proba = weather_aus["model"].predict_proba(input_data)

    if prediction[0] == 1:
        st.error(
            f"Prediction: It will rain tomorrow with a probability of {prediction_proba[0][1] * 100:.1f}%"
        )
    else:
        st.success(
            f"Prediction: It will not rain tomorrow with a probability of {prediction_proba[0][0] * 100:.1f}%"
        )
