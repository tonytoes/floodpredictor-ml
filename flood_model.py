#!/usr/bin/env python
# coding: utf-8
"""
flood_model.py
Core model functions — imported by flood_dashboard.py
"""

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
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv(".env")

API_KEY  = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# ── Cities ────────────────────────────────────────────────────────────────────
cities = {
    "Quezon City": (14.65, 121.03),
    "Manila":      (14.60, 120.98),
    "Marikina":    (14.65, 121.10),
    "Pasig":       (14.58, 121.06),
}


# ── Weather (Open-Meteo, no key needed) ───────────────────────────────────────
def get_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "precipitation,temperature_2m,relativehumidity_2m",
        "timezone":  "Asia/Manila",
    }
    response = requests.get(url, params=params)
    data = response.json()

    hourly   = data["hourly"]
    rain     = hourly["precipitation"]
    temp     = hourly["temperature_2m"]
    humidity = hourly["relativehumidity_2m"]

    rain_24h     = sum(rain[:24])
    avg_temp     = sum(temp[:24]) / 24
    avg_humidity = sum(humidity[:24]) / 24

    return rain_24h, avg_temp, avg_humidity


# ── Data preparation ──────────────────────────────────────────────────────────
def prepare_data(data):
    le = LabelEncoder()
    data["Location"] = le.fit_transform(data["Location"])
    data["Month"] = pd.to_datetime(data["Date"]).dt.month
    data["Day"]   = pd.to_datetime(data["Date"]).dt.day

    X = data[["Month", "Day", "Location",
              "Rainfall_mm", "WaterLevel_m",
              "SoilMoisture_pct", "Elevation_m"]]
    y = data["FloodOccurrence"]
    return X, y, le


# ── EDA (standalone matplotlib windows — kept for CLI use) ────────────────────
def eda_analysis(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="FloodOccurrence")
    plt.title("Flood Occurrence Count")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["Rainfall_mm"], kde=True)
    plt.title("Distribution of Rainfall (mm)")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, y="WaterLevel_m")
    plt.title("Water Level (m) Box Plot")
    plt.show()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 4:
        plt.figure(figsize=(8, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap of Numeric Features")
        plt.show()

    sns.pairplot(numeric_df)
    plt.show()


# ── Classifier ────────────────────────────────────────────────────────────────
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    return model, acc


# ── Regression helpers ────────────────────────────────────────────────────────
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


def predict_future(model, current_value, steps=5):
    predictions = [current_value]
    for _ in range(steps):
        nxt = model.predict(np.array([[predictions[-1]]]))
        predictions.append(nxt[0])
    return predictions[1:]


# ── Per-city flood risk (uses live Open-Meteo weather) ───────────────────────
def predict_flood_risk(model, le, city_name, lat, lon):
    rain, temp, humidity = get_weather_data(lat, lon)
    now = datetime.now()

    location_encoded = le.transform([city_name])[0]

    water_level   = 2.0
    soil_moisture = humidity   # approximation
    elevation     = 10.0

    input_data = pd.DataFrame([{
        "Month":            now.month,
        "Day":              now.day,
        "Location":         location_encoded,
        "Rainfall_mm":      rain,
        "WaterLevel_m":     water_level,
        "SoilMoisture_pct": soil_moisture,
        "Elevation_m":      elevation,
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    return prediction, prob, rain, temp, humidity


def display_flood_risks(model, le):
    results = {}
    for city, (lat, lon) in cities.items():
        try:
            pred, prob, rain, temp, humidity = predict_flood_risk(
                model, le, city, lat, lon
            )
            status = "HIGH FLOOD RISK" if pred == 1 else "LOW FLOOD RISK"
            print(f"{city}: {status}  ({prob*100:.1f}%)")
            results[city] = {
                "prediction":  int(pred),
                "probability": float(prob),
                "rain_24h":    rain,
                "avg_temp":    temp,
                "avg_humidity": humidity,
            }
        except Exception as e:
            print(f"{city}: Error — {e}")
            results[city] = {"error": str(e)}
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("dataset/flood-prediction.csv")
    X, y, le = prepare_data(df)
    model, _ = train_model(X, y)
    display_flood_risks(model, le)
