import pandas as pd
import json


data = {
    "data": {
        "request": [
            {
                "type": "City",
                "query": "Brussels, Belgium"
            }
        ],
        "weather": [
            {
                "date": "2021-01-01",
                "astronomy": [
                    {
                        "sunrise": "08:45 AM",
                        "sunset": "04:48 PM",
                        "moonrise": "07:11 PM",
                        "moonset": "10:42 AM",
                        "moon_phase": "Waning Gibbous",
                        "moon_illumination": "79"
                    }
                ],
                "maxtempC": "5",
                "maxtempF": "40",
                "mintempC": "1",
                "mintempF": "33",
                "avgtempC": "2",
                "avgtempF": "36",
                "totalSnow_cm": "0.0",
                "sunHour": "4.9",
                "uvIndex": "1",
                "hourly": [
                    {
                        "time": "24",
                        "tempC": "5",
                        "tempF": "40",
                        "windspeedMiles": "4",
                        "windspeedKmph": "7",
                        "winddirDegree": "224",
                        "winddir16Point": "SW",
                        "weatherCode": "122",
                        "weatherIconUrl": [
                            {
                                "value": "https://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0004_black_low_cloud.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Overcast"
                            }
                        ],
                        "precipMM": "0.1",
                        "precipInches": "0.0",
                        "humidity": "90",
                        "visibility": "7",
                        "visibilityMiles": "4",
                        "pressure": "1010",
                        "pressureInches": "30",
                        "cloudcover": "70",
                        "HeatIndexC": "2",
                        "HeatIndexF": "36",
                        "DewPointC": "1",
                        "DewPointF": "34",
                        "WindChillC": "1",
                        "WindChillF": "33",
                        "WindGustMiles": "8",
                        "WindGustKmph": "13",
                        "FeelsLikeC": "1",
                        "FeelsLikeF": "33",
                        "uvIndex": "1"
                    }
                ]
            },
            {
                "date": "2021-01-02",
                "astronomy": [
                    {
                        "sunrise": "08:45 AM",
                        "sunset": "04:49 PM",
                        "moonrise": "08:27 PM",
                        "moonset": "11:13 AM",
                        "moon_phase": "Waning Gibbous",
                        "moon_illumination": "71"
                    }
                ],
                "maxtempC": "4",
                "maxtempF": "38",
                "mintempC": "0",
                "mintempF": "32",
                "avgtempC": "1",
                "avgtempF": "35",
                "totalSnow_cm": "0.0",
                "sunHour": "3.0",
                "uvIndex": "1",
                "hourly": [
                    {
                        "time": "24",
                        "tempC": "4",
                        "tempF": "38",
                        "windspeedMiles": "5",
                        "windspeedKmph": "8",
                        "winddirDegree": "234",
                        "winddir16Point": "SW",
                        "weatherCode": "122",
                        "weatherIconUrl": [
                            {
                                "value": "https://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0004_black_low_cloud.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Overcast"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "87",
                        "visibility": "9",
                        "visibilityMiles": "5",
                        "pressure": "1014",
                        "pressureInches": "30",
                        "cloudcover": "66",
                        "HeatIndexC": "1",
                        "HeatIndexF": "35",
                        "DewPointC": "-0",
                        "DewPointF": "32",
                        "WindChillC": "-1",
                        "WindChillF": "31",
                        "WindGustMiles": "8",
                        "WindGustKmph": "13",
                        "FeelsLikeC": "-1",
                        "FeelsLikeF": "31",
                        "uvIndex": "1"
                    }
                ]
            }
        ]
    }
}

weather_data = data['data']['weather']

# Create an empty list to store the selected data
selected_data = []

# Iterate through the "weather" entries
for entry in weather_data:
    date = entry['date']
    for hour_entry in entry['hourly']:
        selected_row = {
            'date': date,
            'maxtempC': hour_entry['tempC'],
            'mintempC': hour_entry['tempC'],
            'avgtempC': hour_entry['tempC'],
            'sunHour': hour_entry['sunHour'],
            'uvIndex': hour_entry['uvIndex'],
            'FeelsLikeC': hour_entry['FeelsLikeC'],
            'humidity': hour_entry['humidity'],
            'windspeedKmph': hour_entry['windspeedKmph']
        }
        selected_data.append(selected_row)

# Create a DataFrame from the selected data
selected_df = pd.DataFrame(selected_data)

# Display the DataFrame
print(selected_data.head)