# modules/crop_yield_predictor.py
import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
from .weather_service import WeatherService

class CropYieldPredictor:
    def __init__(self):
        self._load_models()
        self.weather_service = WeatherService()
        self.column_order = [
            "Crop",
            "Precipitation (mm day-1)",
            "Specific Humidity at 2 Meters (g/kg)",
            "Relative Humidity at 2 Meters (%)",
            "Temperature at 2 Meters (C)"
        ]

    def _load_models(self):
        try:
            self.model = joblib.load("models/random_forest_model.pkl")
            self.label_encoder = joblib.load("models/crop_encoder.pkl")
            self.scaler = joblib.load("models/crop_scaler.pkl")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def predict(self, city_name: str, crop_type: str, days: int = 30) -> Optional[float]:
        try:
            coordinates = self.weather_service.get_coordinates(city_name)
            if not coordinates:
                return None

            weather_data = self._get_weather_data(coordinates, days)
            if weather_data is None:
                return None

            return self._make_prediction(weather_data, crop_type)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def _get_weather_data(self, coordinates: Tuple[float, float], days: int) -> Optional[pd.DataFrame]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.weather_service.fetch_weather_data(
            coordinates[0], coordinates[1],
            start_date.strftime('%Y%m%d'),
            end_date.strftime('%Y%m%d')
        )

    def _make_prediction(self, weather_data: pd.DataFrame, crop_type: str) -> float:
        crop_encoded = self.label_encoder.transform([crop_type])[0]
        weather_data.insert(0, 'Crop', crop_encoded)
        
        weather_data = weather_data[self.column_order]
        numerical_columns = self.column_order[1:]
        weather_data[numerical_columns] = self.scaler.transform(weather_data[numerical_columns])
        
        return self.model.predict(weather_data)[0]

    def get_available_crops(self) -> list:
        return list(self.label_encoder.classes_)