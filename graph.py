#%%
import numpy as np
import pandas as pd
import networkx as nx

data = pd.read_csv('dataset/Airlines.csv')

data.shape
(100, 16)

data.dtypes

# converting sched_dep_time to 'std' - Scheduled time of departure
data['std'] = data.sched_dep_time.astype(str).str.replace('(\d{2}$)', '') + ':' + data.sched_dep_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
# converting sched_arr_time to 'sta' - Scheduled time of arrival
data['sta'] = data.sched_arr_time.astype(str).str.replace(r'(\d{2}$)', '') + ':' + data.sched_arr_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
# converting dep_time to 'atd' Actual time of departure
data['atd'] = data.dep_time.fillna(0).astype(np.int64).astype(str).str.replace(r'(\d{2}$)', '') + ':' + data.dep_time.fillna(0).astype(np.int64).astype(str).str.extract('(\d{2}$)', expand=False) + ':00'

# converting arr_time to 'ata' - Actual time of arrival
data['ata'] = data.arr_time.fillna(0).astype (np.int64).astype(str).str.replace(r'(\d{2}$)', '') + ':' + data.arr_time.fillna(0).astype(np.int64).astype(str).str.extract('(\d{2}$)', expand=False) + ':00'

data['date'] = pd.to_datetime(data[['year', 'month', 'day']])

data = data.drop(columns = ['year', 'month', 'day'])
FG = nx.from_pandas_edgelist (data, source='origin', target='dest', edge_attr =True,)
FG.nodes()
FG.edges()

nx.draw_networkx(FG, with_labels=True)
nx.algorithms.degree_centrality(FG)
nx.density(FG)
nx.average_shortest_path_length(FG)
nx.average_degree_connectivity(FG)

dijpath = nx.dijkstra_path(FG, source='JAX', target='DFW')

shortpath = nx.shortest_path(FG, source='JAX', target='DFW', weight='air_time')
# %%
