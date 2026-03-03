#%%
import numpy as np
import pandas as pd

data = pd.read_csv('dataset/Airlines.csv')

data.shape
(100, 16)

data.dtypes

# converting sched_dep_time to 'std' - Scheduled time of departure
data['std'] = data.sched_dep_time.astype(str).str.replace('(\d{2}$)', '') + ':' + data.sched_dep_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
# %%
