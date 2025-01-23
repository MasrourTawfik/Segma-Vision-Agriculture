# modules/weather_service.py
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import requests
import pandas as pd
from typing import Tuple, Optional, Dict
import time

class WeatherService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="agricultural_analysis_app")
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def get_coordinates(self, city_name: str, max_retries: int = 3) -> Optional[Tuple[float, float]]:
        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(city_name)
                if location:
                    return location.latitude, location.longitude
                return None
            except GeocoderTimedOut:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
                
    def fetch_weather_data(self, latitude: float, longitude: float, 
                          start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        params = {
            "start": start_date,
            "end": end_date,
            "latitude": latitude,
            "longitude": longitude,
            "community": "AG",
            "parameters": "T2M,RH2M,PRECTOTCORR,QV2M",
            "format": "JSON"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return self._process_weather_data(response.json())
        except requests.exceptions.RequestException:
            return None
            
    def _process_weather_data(self, data: Dict) -> pd.DataFrame:
        parameters = data['properties']['parameter']
        daily_data = []
        
        for date in parameters['T2M'].keys():
            daily_values = {
                'Temperature at 2 Meters (C)': parameters['T2M'][date],
                'Precipitation (mm day-1)': parameters['PRECTOTCORR'][date],
                'Specific Humidity at 2 Meters (g/kg)': parameters['QV2M'][date],
                'Relative Humidity at 2 Meters (%)': parameters['RH2M'][date]
            }
            daily_data.append(daily_values)
            
        return pd.DataFrame(daily_data).mean().to_frame().T