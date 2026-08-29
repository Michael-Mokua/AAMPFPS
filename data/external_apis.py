import requests
import logging
import yaml
import os
import time

logger = logging.getLogger(__name__)

class ExternalAPIClient:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path) and os.path.exists("../config.yaml"):
            config_path = "../config.yaml"
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.keys = self.config.get('api_keys', {})

    def get_weather(self, lat, lon, date):
        """
        Fetches weather. Primary: Open-Meteo. Fallback: OpenWeatherMap.
        """
        try:
            # Primary: Open-Meteo
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,windspeed_10m&start_date={date}&end_date={date}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data['hourly']['temperature_2m'][12],
                    "precipitation": data['hourly']['precipitation'][12],
                    "wind_speed": data['hourly']['windspeed_10m'][12],
                    "source": "open-meteo"
                }
        except Exception as e:
            logger.warning(f"Open-meteo failed. Attempting fallback: {e}")
            
        try:
            # Fallback: OpenWeatherMap
            owm_key = self.keys.get('open_weather')
            if owm_key and owm_key != "PLACEHOLDER_WEATHER_API_KEY":
                owm_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={owm_key}&units=metric"
                response = requests.get(owm_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Just taking the first forecast block for approximation
                    return {
                        "temperature": data['list'][0]['main']['temp'],
                        "precipitation": data['list'][0].get('rain', {}).get('3h', 0.0),
                        "wind_speed": data['list'][0]['wind']['speed'] * 3.6, # m/s to km/h
                        "source": "openweathermap"
                    }
        except Exception as e:
            logger.error(f"Weather API Exception: {e}")
            
        return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 10.0, "source": "fallback"}

    def get_api_football_data(self, endpoint, params=None):
        """
        Generic fetcher for API-Football endpoints (lineups, fixtures, injuries).
        """
        key = self.keys.get('api_football')
        if not key or key == "PLACEHOLDER_API_FOOTBALL_KEY":
            logger.warning("Missing API-Football Key.")
            return None
            
        headers = {
            'x-apisports-key': key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        
        url = f"https://v3.football.api-sports.io/{endpoint}"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()['response']
            else:
                logger.error(f"API-Football Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"API-Football request failed: {e}")
            return None

    def get_sportmonks_data(self, endpoint, params=None):
        """
        Generic fetcher for Sportmonks (Referees, News).
        """
        key = self.keys.get('sportsmonks')
        if not key or key == "PLACEHOLDER_SPORTSMONKS_KEY":
            logger.warning("Missing Sportmonks Key. Returning []")
            return []
            
        url = f"https://api.sportmonks.com/v3/football/{endpoint}"
        if params is None:
            params = {}
        params['api_token'] = key
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()['data']
            return []
        except Exception as e:
            logger.error(f"Sportmonks request failed: {e}")
            return []
