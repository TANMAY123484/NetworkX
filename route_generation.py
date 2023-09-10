# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:00:05 2023

@author: Admin
"""

import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict
import numpy as np
import fiona

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
def get_direct_distance(node1, node2):
    return euclidean_distance_lookup.get((node1, node2), float('inf'))

def adjust_edge_weights_with_circuity(G, lambda_penalty):
    for origin, destination, data in G.edges(data=True):
        direct_distance = get_direct_distance(origin, destination)
        circuity = (data['total_cost'] - direct_distance) / direct_distance if direct_distance else float('inf')
        data['penalized_cost'] = data['total_cost'] * (1 + lambda_penalty * circuity)

def construct_feeder_lines_graph_with_circuity(G, lambda_penalty):
    adjust_edge_weights_with_circuity(G, lambda_penalty)
    return construct_feeder_lines_graph_refined_with_prioritization(G, weight='penalized_cost')

def construct_feeder_lines_graph_without_circuity(G):
    return construct_feeder_lines_graph_refined_with_prioritization(G, weight='total_cost')

used_edges = defaultdict(int)

def construct_feeder_lines_graph_refined_with_prioritization(G, weight):
    lines = []
    visited_nodes = set()

    def get_starting_node():
        unvisited = list(set(G.nodes()) - visited_nodes)
        if unvisited:
            sorted_nodes = sorted(unvisited, key=lambda x: G.degree(x), reverse=True)
            return sorted_nodes[0]
        return None

    current_node = get_starting_node()

    while current_node:
        current_line = [current_node]
        visited_nodes.add(current_node)

        while len(current_line) < MAX_NODES_PER_LINE:
            neighbors = [(neighbor, G[current_node][neighbor][weight], used_edges[(current_node, neighbor)])
                         for neighbor in G.neighbors(current_node)
                         if neighbor not in visited_nodes]

            if not neighbors:
                break

            neighbors.sort(key=lambda x: (x[2], x[1]))  # Sorting by least used edge and then by distance
            next_node = neighbors[0][0]
            used_edges[(current_node, next_node)] += 1

            current_node = next_node
            current_line.append(current_node)
            visited_nodes.add(current_node)

            if len(current_line) > MIN_NODES_PER_LINE and random.random() < 0.2:
                break

        lines.append(current_line)
        current_node = get_starting_node()

    return lines

node_coordinates = {row['id']: (row['lat'], row['long']) for _, row in df3.iterrows()}

def plot_feeder_lines_with_vs_without_circuity(G, lambda_penalty, node_coordinates):
    # Get feeder lines
    feeder_lines_with_circuity = construct_feeder_lines_graph_with_circuity(G, lambda_penalty)
    feeder_lines_without_circuity = construct_feeder_lines_graph_without_circuity(G)

    # Convert routes to GeoDataFrames
    gdf_with_circuity = convert_routes_to_gdf(feeder_lines_with_circuity)
    gdf_without_circuity = convert_routes_to_gdf(feeder_lines_without_circuity)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf_with_circuity.plot(ax=ax, color="blue", linewidth=2, label="With Circuity")
    gdf_without_circuity.plot(ax=ax, color="green", linewidth=2, linestyle='--', label="Without Circuity")

    # Plot nodes
    gdf_nodes = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in node_coordinates.values()],
        crs="EPSG:32643"
    )
    gdf_nodes['node_id'] = list(node_coordinates.keys())
    gdf_nodes.plot(ax=ax, markersize=5, color="red", label="Nodes")

    ax.set_title("Feeder Lines With vs Without Circuity")
    plt.legend(loc="upper left")
    plt.show()

def convert_routes_to_gdf(feeder_lines):
    # Convert the feeder lines to LineString objects, but only if they have at least two nodes
    lines = [LineString([node_coordinates[node] for node in line]) for line in feeder_lines if len(line) > 1]
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:32643")
    gdf['geometry'] = gdf['geometry'].simplify(0.001)
    return gdf

# Get feeder lines
feeder_lines_with_circuity = construct_feeder_lines_graph_with_circuity(G, lambda_penalty=0.5)
feeder_lines_without_circuity = construct_feeder_lines_graph_without_circuity(G)

# Convert routes to GeoDataFrames
gdf_with_circuity = convert_routes_to_gdf(feeder_lines_with_circuity)
gdf_without_circuity = convert_routes_to_gdf(feeder_lines_without_circuity)

# Save the GeoDataFrames as shapefiles
gdf_with_circuity.to_file("metro_feeder_routes_with_circuity.shp3",encoding='utf-8')
gdf_without_circuity.to_file("metro_feeder_routes_without_circuity.shp3",encoding='utf-8')

# Call the plot function with the graph G and the lambda_penalty you wish to use
plot_feeder_lines_with_vs_without_circuity(G, lambda_penalty=0.5, node_coordinates=node_coordinates)