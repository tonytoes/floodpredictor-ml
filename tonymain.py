import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")


API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


def get_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,temperature_2m,relativehumidity_2m",
        "timezone": "Asia/Manila"
    }

    response = requests.get(url, params=params)
    data = response.json()

    hourly = data['hourly']

    rain = hourly['precipitation']
    temp = hourly['temperature_2m']
    humidity = hourly['relativehumidity_2m']

    rain_24h = sum(rain[:24])
    avg_temp = sum(temp[:24]) / 24
    avg_humidity = sum(humidity[:24]) / 24

    return rain_24h, avg_temp, avg_humidity
  
cities = {
    "Quezon City": (14.65, 121.03),
    "Manila": (14.60, 120.98),
    "Marikina": (14.65, 121.10),
    "Pasig": (14.58, 121.06),
    }
  
def prepare_data(data):
  le =  LabelEncoder()
  data['Location'] = le.fit_transform(data['Location'])
  data['Month'] = pd.to_datetime(data['Date']).dt.month
  data['Day'] = pd.to_datetime(data['Date']).dt.day
  
  X = data[['Month', 'Day', 'Location', 'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m']]
  y = data['FloodOccurrence']
  
  return X, y, le

def eda_analysis(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Count plot for FloodOccurrence
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='FloodOccurrence')
    plt.title('Flood Occurrence Count')
    plt.xlabel('Flood Occurrence')
    plt.ylabel('Count')
    plt.show()

    # Histogram of Rainfall
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Rainfall_mm'], kde=True)
    plt.title('Distribution of Rainfall (mm)')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.show()

    # Box plot for Water Level
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, y='WaterLevel_m')
    plt.title('Water Level (m) Box Plot')
    plt.ylabel('Water Level (m)')
    plt.show()

    # Correlation heatmap for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 4:
        plt.figure(figsize=(8, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Numeric Features')
        plt.show()

    # Pair plot for numeric features
    sns.pairplot(numeric_df)
    plt.show()
    
def train_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  print("Accuracy:", accuracy_score(y_test, y_pred))

  return model
   
def regression_data(data, feature):
  X, y = [], []
  
  for i in range(len(data) - 1):
    X.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i + 1])
    
  X = np.array(X).reshape(-1, 1)
  y = np.array(y)
  return X, y

def train_regression_model(X, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, y)
  
  return model

def predict_flood_risk(model, le, city_name, lat, lon):
    from datetime import datetime

    rain, temp, humidity = get_weather_data(lat, lon)

    now = datetime.now()

    # Encode location
    location_encoded = le.transform([city_name])[0]

    # ⚠️ Placeholder values (you can improve later)
    water_level = 2.0
    soil_moisture = humidity  # approximation
    elevation = 10.0

    input_data = pd.DataFrame([{
        'Month': now.month,
        'Day': now.day,
        'Location': location_encoded,
        'Rainfall_mm': rain,
        'WaterLevel_m': water_level,
        'SoilMoisture_pct': soil_moisture,
        'Elevation_m': elevation
    }])

    prediction = model.predict(input_data)[0]

    return prediction

def display_flood_risks(model, le):
    for city, (lat, lon) in cities.items():
        try:
            risk = predict_flood_risk(model, le, city, lat, lon)

            if risk == 1:
                status = "HIGH FLOOD RISK"
            else:
                status = "LOW FLOOD RISK"

            print(f"{city}: {status}")

        except Exception as e:
            print(f"{city}: Error - {e}")
            
            
df = pd.read_csv("dataset/flood-prediction.csv")

X, y, le = prepare_data(df)
model = train_model(X, y)

display_flood_risks(model, le)