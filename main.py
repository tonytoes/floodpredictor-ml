#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")


# In[ ]:


API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


# In[ ]:


def get_current_weather(city):
  url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
  response = requests.get(url)
  data = response.json()
  return {
    'city': data['name'],
    'current_temp':round(data['main']['temp'])
  }


# In[ ]:


def prepare_data(data):
  le =  LabelEncoder()
  data['Location'] = le.fit_transform(data['Location'])
  data['Month'] = pd.to_datetime(data['Date']).dt.month
  data['Day'] = pd.to_datetime(data['Date']).dt.day

  X = data[['Month', 'Day', 'Location', 'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m']]
  y = data['FloodOccurrence']

  return X, y, le


# In[ ]:


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


# In[ ]:


def train_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

  return model


# In[ ]:


def regression_data(data, feature):
  X, y = [], []

  for i in range(len(data) - 1):
    X.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i + 1])

  X = np.array(X).reshape(-1, 1)
  y = np.array(y)
  return X, y


# In[ ]:


def train_regression_model(X, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, y)

  return model


# In[ ]:


def predict_future(model, current_value):
  predictions = [current_value]

  for i in range(5):
    next_value = model.predict(np.array([[predictions[-1]]]))

    predictions.append(next_value[0])

  return predictions[1:]

