# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 23:50:27 2023

@author: Admin
"""

import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict

# Load datasets
df_new = pd.read_csv("new_snapped_od_matrix_allocated_feeder.csv")
df_new = df_new.iloc[:, [0, 1, 5]]
df_new.fillna(0, inplace=True)
df_new.columns = ['origin_id', 'destination_id', 'total_cost']
df_new.loc[df_new['origin_id'] == df_new['destination_id'], 'total_cost'] = 0

df2 = pd.read_csv("new_euclidean_allocated_feeder.csv")
df2.columns = ['origin_id', 'destination_id', 'total_cost']

df3 = pd.read_csv("reprojected_allocated_feeder_stops_new.csv")

euclidean_distance_lookup = {tuple(row[['origin_id', 'destination_id']]): row['total_cost'] for _, row in df2.iterrows()}

G = nx.from_pandas_edgelist(df_new, 'origin_id', 'destination_id', ['total_cost'], create_using=nx.DiGraph())

# Create adjacency list
adjacency_list = defaultdict(list)
for edge in G.edges():
    adjacency_list[edge[0]].append(edge[1])

# Depth First Search
visited = set()
order_of_visit = []

def DFS_iterative(node):
    stack = [node]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            order_of_visit.append(vertex)
            visited.add(vertex)
            stack.extend(list(set(adjacency_list[vertex]) - visited))

start_node = list(G.nodes())[0]
DFS_iterative(start_node)

MAX_NODES_PER_LINE = 10
MIN_NODES_PER_LINE = 5

# Function to calculate direct distance between two nodes
def get_direct_distance(node1, node2):
    return euclidean_distance_lookup.get((node1, node2), float('inf'))

# Calculate penalized cost for edges
def adjust_edge_weights_with_circuity(G, lambda_penalty):
    for origin, destination, data in G.edges(data=True):
        direct_distance = get_direct_distance(origin, destination)
        circuity = (data['total_cost'] - direct_distance) / direct_distance if direct_distance else float('inf')
        data['penalized_cost'] = data['total_cost'] * (1 + lambda_penalty * circuity)

# Function to construct feeder lines using TSP
def construct_feeder_lines_with_tsp(G, lambda_penalty):
    adjust_edge_weights_with_circuity(G, lambda_penalty)
    feeder_lines = []
    visited_nodes = set()
    
    while visited_nodes != set(G.nodes()):
        unvisited_nodes = list(set(G.nodes()) - visited_nodes)
        current_node = random.choice(unvisited_nodes)
        tsp_path = nx.approximation.traveling_salesman_problem(G, source=current_node)
        
        feeder_lines.append(tsp_path)
        visited_nodes.update(tsp_path)
    
    return feeder_lines

# Get feeder lines using TSP-based approach
feeder_lines_with_tsp = construct_feeder_lines_with_tsp(G, lambda_penalty=0.5)