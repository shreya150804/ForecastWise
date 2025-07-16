import streamlit as st
import requests
import datetime
import pandas as pd
from meteostat import Point, Daily
from prophet import Prophet
import matplotlib.pyplot as plt
import pytz


india = pytz.timezone('Asia/Kolkata')


API_KEY = st.secrets["API_KEY"]



CITY_COORDS = {
    "Pune": (18.52, 73.85),
    "Mumbai": (19.07, 72.88),
    "Delhi": (28.61, 77.20),
    "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.38, 78.48),
    "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27)
}

def get_weather(city):
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(base_url)
    return response.json()

def get_historical_data(lat, lon):
    start = datetime.datetime(2018, 1, 1)
    end = datetime.datetime.now() - datetime.timedelta(days=1)
    location = Point(lat, lon)
    df = Daily(location, start, end).fetch().reset_index()[["time", "tavg"]].dropna()
    df.columns = ["ds", "y"]
    return df

def train_and_forecast(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(5)  # Last 5 predictions


st.set_page_config(page_title="Weather Forecast Dashboard", layout="centered")


# # -- SIDEBAR --
st.sidebar.title("🌐 Select City")
city = st.sidebar.selectbox("Choose a city", list(CITY_COORDS.keys()))
lat, lon = CITY_COORDS[city]

# -- MAIN UI --
st.title("🌦️ ForecastWise: Real-Time + 5-Day Weather Forecast")
st.markdown("Get real-time weather updates and smart temperature predictions using AI.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 City")
    st.markdown(f"**{city}**")

with col2:
    st.subheader("📅 Date")
    st.markdown(f"**{datetime.datetime.now().strftime('%A, %d %B %Y')}**")

with col3:
    st.subheader("⏱️ Time")
    st.markdown(f"**{datetime.datetime.now().strftime('%I:%M %p')}**")


# -- GET DATA --
with st.spinner("Fetching weather info..."):
    weather = get_weather(city)
    if weather.get("cod") != 200:
        st.error("Could not fetch weather data.")
    else:
        temp = weather["main"]["temp"]
        humid = weather["main"]["humidity"]
        wind = weather["wind"]["speed"]
        desc = weather["weather"][0]["description"]

        st.markdown("---")
        st.subheader("🌡️ Current Weather")
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature (°C)", f"{temp} °C")
        col2.metric("Humidity (%)", f"{humid} %")
        col3.metric("Wind Speed (m/s)", f"{wind}")

        st.markdown(f"**Description:** {desc.title()}")
        st.markdown("---")


# -- FORECAST --
with st.spinner("Loading..."):
    hist_df = get_historical_data(lat, lon)
    forecast_df = train_and_forecast(hist_df)

    plt.style.use("dark_background")
    st.subheader("📈 5-Day Forecast (Avg Temperature)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_df["ds"], forecast_df["yhat"], color="skyblue", marker="o", linewidth=2)
    ax.set_title(f"Temperature Forecast for {city}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.dataframe(forecast_df.set_index("ds").rename(columns={"yhat": "Forecast Temp (°C)"}))

# -- FOOTER --
st.markdown("---")
st.caption("🔍 Powered by Meteostat, OpenWeatherMap & Prophet | Built by Shreya Wani")