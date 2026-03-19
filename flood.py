# flood_functions.py

import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

load_dotenv(".env")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp'])
    }

def prepare_data(data):
    le = LabelEncoder()
    data['Location'] = le.fit_transform(data['Location'])
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    data['Day'] = pd.to_datetime(data['Date']).dt.day
    X = data[['Month', 'Day', 'Location', 'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m']]
    y = data['FloodOccurrence']
    return X, y, le

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
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

def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]