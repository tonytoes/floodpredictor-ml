#%%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

file_path ='dataset/flood-prediction.csv'

df = pd.read_csv(file_path, encoding='ascii', delimiter=',')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print(df.head())
#%%