import requests
import json
import os

def get_fixtures():
    api_key = "7fd73a463398e1df8bf27e8b6e8a5cbe"
    headers = {
        'x-apisports-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    # Fetch next 20 fixtures for Premier League (League ID 39)
    url = "https://v3.football.api-sports.io/fixtures?league=39&season=2025&next=20"
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            with open("fixtures_debug.json", "w") as f:
                json.dump(data, f, indent=4)
            print(f"Successfully fetched {len(data.get('response', []))} fixtures.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    get_fixtures()
