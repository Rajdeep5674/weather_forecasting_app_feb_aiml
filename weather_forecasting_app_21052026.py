import streamlit as st
from datetime import date, timedelta
import requests
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def geocode_city(name):
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()

    if not data.get("results"):
        raise ValueError(f"Could not find city: {name}")

    res = data["results"][0]

    return {
        "name": res.get("name"),
        "latitude": res["latitude"],
        "longitude": res["longitude"],
        "timezone": res.get("timezone", "auto"),
        "country": res.get("country"),
        "admin1": res.get("admin1"),
    }


def fetch_history(lat, lon, start_date, end_date, timezone="auto"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": timezone,
    }

    r = requests.get(HIST_URL, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    daily = data.get("daily", {})

    if not daily or "time" not in daily:
        raise RuntimeError("Historical data not available.")

    df = pd.DataFrame(daily)

    df["temperature_2m_max"] = pd.to_numeric(
        df["temperature_2m_max"], errors="coerce"
    )
    df["temperature_2m_min"] = pd.to_numeric(
        df["temperature_2m_min"], errors="coerce"
    )

    df["temp_mean"] = (
        df["temperature_2m_max"] + df["temperature_2m_min"]
    ) / 2

    df["date"] = pd.to_datetime(df["time"])

    df = df.dropna(subset=["temp_mean"]).reset_index(drop=True)

    return df[[
        "date",
        "temperature_2m_min",
        "temperature_2m_max",
        "temp_mean"
    ]]


def fetch_forecast(lat, lon, timezone="auto"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": 7,
        "timezone": timezone,
    }

    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()
    daily = data.get("daily", {})

    if not daily or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily)

    df["date"] = pd.to_datetime(df["time"])
    df["temperature_2m_max"] = pd.to_numeric(
        df["temperature_2m_max"], errors="coerce"
    )
    df["temperature_2m_min"] = pd.to_numeric(
        df["temperature_2m_min"], errors="coerce"
    )

    df["temp_mean"] = (
        df["temperature_2m_max"] + df["temperature_2m_min"]
    ) / 2

    return df[[
        "date",
        "temperature_2m_min",
        "temperature_2m_max",
        "temp_mean"
    ]]


def build_xy(df):
    df = df.sort_values("date").reset_index(drop=True)

    base_date = df["date"].min()

    df["x"] = (df["date"] - base_date).dt.days.astype(int)

    X = df[["x"]].values
    y = df["temp_mean"].values.astype(float)

    return df, X, y, base_date


def fit_poly_regression(X, y, degree=3):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])

    model.fit(X, y)

    return model


st.set_page_config(
    page_title="Weather Temperature Prediction",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Temperature Prediction App")
st.write("Predict tomorrow's mean temperature using Polynomial Regression.")

cities = ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bengaluru", "Hyderabad"]

city = st.selectbox("Choose a city", cities)

degree = st.slider(
    "Select Polynomial Degree",
    min_value=1,
    max_value=5,
    value=3
)

days = st.slider(
    "Historical days for training",
    min_value=30,
    max_value=365,
    value=120
)

if st.button("Predict Tomorrow's Temperature"):

    try:
        with st.spinner("Fetching location data..."):
            place = geocode_city(city)

        lat = place["latitude"]
        lon = place["longitude"]
        timezone = place["timezone"]

        today = date.today()
        start_date = today - timedelta(days=days)
        hist_end = today - timedelta(days=3)
        tomorrow = today + timedelta(days=1)

        with st.spinner("Fetching historical weather data..."):
            hist_df = fetch_history(
                lat,
                lon,
                start_date,
                hist_end,
                timezone
            )

        if hist_df.empty or len(hist_df) < 5:
            st.error("Not enough historical data available.")
        else:
            hist_df, X, y, base_date = build_xy(hist_df)

            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)

            X = X[mask]
            y = y[mask]
            hist_df = hist_df.loc[mask].reset_index(drop=True)

            if len(y) < 5:
                st.error("Too few clean records to train the model.")
            else:
                model = fit_poly_regression(X, y, degree=degree)

                x_tomorrow = np.array([[
                    (pd.Timestamp(tomorrow) - base_date).days
                ]])

                y_pred = float(model.predict(x_tomorrow)[0])

                with st.spinner("Fetching Open-Meteo forecast..."):
                    fc_df = fetch_forecast(lat, lon, timezone)

                fc_val = None

                if not fc_df.empty:
                    tomorrow_data = fc_df.loc[
                        fc_df["date"].dt.date == tomorrow,
                        "temp_mean"
                    ]

                    if not tomorrow_data.empty:
                        fc_val = float(tomorrow_data.iloc[0])

                location_parts = []

                if place.get("name"):
                    location_parts.append(place["name"])
                if place.get("admin1"):
                    location_parts.append(place["admin1"])
                if place.get("country"):
                    location_parts.append(place["country"])

                location = ", ".join(location_parts)

                st.success("Prediction completed!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📍 Location", location)

                with col2:
                    st.metric(
                        "🤖 Model Prediction",
                        f"{y_pred:.2f} °C"
                    )

                with col3:
                    if fc_val is not None:
                        st.metric(
                            "🌐 Open-Meteo Forecast",
                            f"{fc_val:.2f} °C",
                            delta=f"{y_pred - fc_val:+.2f} °C"
                        )
                    else:
                        st.warning("Forecast unavailable.")

                st.subheader("📊 Historical Weather Data")
                st.dataframe(hist_df)

                st.subheader("📈 Mean Temperature Trend")
                chart_df = hist_df.set_index("date")[["temp_mean"]]
                st.line_chart(chart_df)

                if not fc_df.empty:
                    st.subheader("🔮 Next 7 Days Forecast")
                    st.dataframe(fc_df)
                    st.line_chart(fc_df.set_index("date")[["temp_mean"]])

    except Exception as e:
        st.error(f"Error: {e}")