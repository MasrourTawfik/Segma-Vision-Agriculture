# crop_yield_predictor.py (modules/crop_yield_predictor.py)
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def get_coordinates(city_name, max_retries=3, timeout=5):
    for attempt in range(max_retries):
        try:
            geolocator = Nominatim(user_agent="weather_data_fetcher", timeout=timeout)
            location = geolocator.geocode(city_name)
            if location is None:
                raise ValueError(f"Could not find coordinates for {city_name}")
            return location.latitude, location.longitude
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error finding location: {e}")
                return None
            time.sleep(2)

def fetch_nasa_power_data(latitude, longitude, start_date, end_date):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    parameters = ["T2M", "RH2M", "PRECTOTCORR", "QV2M"]
    
    params = {
        "start": start_date,
        "end": end_date,
        "latitude": latitude,
        "longitude": longitude,
        "community": "AG",
        "parameters": ",".join(parameters),
        "format": "JSON"
    }
    
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def process_weather_data(data):
    if not data:
        return None
        
    try:
        parameter_data = data['properties']['parameter']
        fill_value = data['header']['fill_value']
        daily_data = []
        
        for date in parameter_data['T2M'].keys():
            if parameter_data['T2M'][date] != fill_value:
                daily_values = {
                    'Temperature at 2 Meters (C)': parameter_data['T2M'][date],
                    'Precipitation (mm day-1)': parameter_data['PRECTOTCORR'][date],
                    'Specific Humidity at 2 Meters (g/kg)': parameter_data['QV2M'][date],
                    'Relative Humidity at 2 Meters (%)': parameter_data['RH2M'][date]
                }
                daily_data.append(daily_values)
        
        return pd.DataFrame(daily_data)
    except Exception as e:
        st.error(f"Error processing weather data: {e}")
        return None

def run_prediction():
    st.title("Crop Yield Prediction System")
    
    try:
        model = joblib.load("models/random_forest_model.pkl")
        encoder = joblib.load("models/crop_encoder.pkl")
        scaler = joblib.load("models/crop_scaler.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.text_input("Enter City Name")
        days = st.slider("Analysis Period (days)", 7, 90, 30)
    
    with col2:
        available_crops = list(encoder.classes_)
        crop = st.selectbox("Select Crop", available_crops)
    
    if st.button("Predict Yield"):
        if not city:
            st.warning("Please enter a city name")
            return
            
        with st.spinner("Fetching weather data and predicting yield..."):
            coordinates = get_coordinates(city)
            if coordinates:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                weather_data = fetch_nasa_power_data(
                    coordinates[0], coordinates[1],
                    start_date.strftime('%Y%m%d'),
                    end_date.strftime('%Y%m%d')
                )
                
                if weather_data:
                    daily_weather = process_weather_data(weather_data)
                    if daily_weather is not None:
                        aggregated_weather = daily_weather.mean().to_frame().T
                        
                        crop_encoded = encoder.transform([crop])[0]
                        aggregated_weather.insert(0, 'Crop', crop_encoded)
                        
                        columns = [
                            "Crop",
                            "Precipitation (mm day-1)",
                            "Specific Humidity at 2 Meters (g/kg)",
                            "Relative Humidity at 2 Meters (%)",
                            "Temperature at 2 Meters (C)"
                        ]
                        
                        aggregated_weather = aggregated_weather[columns]
                        numerical_columns = columns[1:]
                        aggregated_weather[numerical_columns] = scaler.transform(aggregated_weather[numerical_columns])
                        
                        prediction = model.predict(aggregated_weather)[0]
                        st.success(f"Predicted Yield: {prediction:.2f} tons/hectare")
                        
                        st.subheader("Weather Summary")
                        st.line_chart(daily_weather)