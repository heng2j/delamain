"""
Shortest Path Networkx Graph and List

This script creates a networkx graph using the topology data previously created.
We have at the moment some hardcoded starting and ending locations for which it find the closest node.
Then, we find the shortest path between the two nodes and show the shortest path on the graph.

Beginning part of the code is based on: https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial

Modified, adapted, and developed by DevGlitch
"""

import os
import itertools
import copy
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial

# Grab edge list data
edgelist = pd.read_parquet('Town02_data/Town02_topology_edge_list.parquet')

# Preview edgelist
# print(edgelist.head(10), "\n")

# Grab node list data hosted on Gist
nodelist = pd.read_parquet('Town02_data/Town02_topology_node_list.parquet')

# Preview nodelist
# print(nodelist.head(5), "\n")

# Create empty graph
g = nx.Graph()

# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())

# Edge list example
# print(elrow[0], "\n")  # node1
# print(elrow[1], "\n")  # node2
# print(elrow[2:].to_dict(), "\n")  # edge attribute dict

# Add node attributes
for i, nlrow in nodelist.iterrows():
    g.nodes[nlrow['id']].update(nlrow[1:].to_dict())

# Node list example
# print(nlrow, "\n")

# Preview first 5 edges
# print(list(g.edges(data=True))[0:5], "\n")

# Preview first 10 nodes
# print(list(g.nodes(data=True))[0:10], "\n")

# Preview total number of edges and nodes
# print('# of edges: {}'.format(g.number_of_edges()), "\n")
# print('# of nodes: {}'.format(g.number_of_nodes()), "\n")

# Define node positions data structure (dict) for plotting
node_positions = {node[0]: (node[1]['lat'], -node[1]['lon']) for node in g.nodes(data=True)}

# Preview of node_positions with a bit of hack (there is no head/slice method for dictionaries).
# print(dict(list(node_positions.items())[0:5]), "\n")

# Making a nice plot that lines up nicely and should look like the carla map
plt.figure(figsize=(8, 6))
nx.draw(
    g,
    pos=node_positions,
    node_size=20,
    node_color="red",
    edge_color="blue"
)

# Display plot
# plt.show()

# Filename using the name of the current carla map running
directory = "Town02_data"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "Town02_networkx_graph.png"
filepath = os.path.join(directory, filename)

# Save Networkx Graph
if os.path.isfile(filepath):
    print("File already exist. No additional networkx graph was saved.")
else:
    print("Saving networkx graph (png file).")
    plt.savefig(filepath)

# Subset dataframe of node list incl. only lat and lon coordinates
node_geo = nodelist[["lat", "lon"]]
# print(node_geo.head(10), "\n")

# Create array for use with spicy
node_geo_array = np.array(node_geo)
# print(node_geo_array)

# Starting and destination locations
start_location = [-0.055, -0.006]  # This is a test value will need to be the GNSS sensor data here
end_location = [10, 20]  # This is a test value will need to be the selected destination here


def find_closest_node(node_list, node_array, location):
    """
    Find the closest node to a geolocation
    Returns the Node ID, Latitude, Longitude, and Altitude.
    """
    distance, index = spatial.KDTree(node_array).query(location)
    return node_list.iloc[index]


# Get closest node to start and end locations
start_location_closest_node = find_closest_node(
    node_list=nodelist,
    node_array=node_geo_array,
    location=start_location)
# print(start_location_closest_node, "\n")

end_location_closest_node = find_closest_node(
    node_list=nodelist,
    node_array=node_geo_array,
    location=end_location)
# print(end_location_closest_node, "\n")

# Compute shortest path between the two nodes closest to start and end locations
# Return a list with node IDs with the first value being the starting node and the last value the ending node.
shortest_path = nx.shortest_path(
    g,
    source=start_location_closest_node[0],
    target=end_location_closest_node[0],
    weight="distance"
)

# See list of nodes of the shortest path
print("Shortest Path:", shortest_path, "\n")

# print(nodelist.head(5), "\n")

# Get lat, lon, and alt attributes of each nodes
shortest_path_geo = pd.DataFrame(columns=["id", "lat", "lon", "alt"])
rows_list = []
for i in shortest_path:
    node_attributes = nodelist.loc[nodelist['id'] == i]
    # print(node_attributes)
    shortest_path_geo = shortest_path_geo.append(node_attributes)

# Append destination to dataframe
destination_attributes = {"id": 999, "lat": 10, "lon": 20, "alt": 0}
shortest_path_geo = shortest_path_geo.append(destination_attributes, ignore_index=True)

# Show shortest path dataframe
print(shortest_path_geo)

# Create a new graph to overlay on the shortest path on it
nx.draw(g, pos=node_positions, node_size=20, edge_color='black', node_color='black')
path = shortest_path
path_edges = list(zip(path, path[1:]))
nx.draw_networkx_nodes(g, pos=node_positions, nodelist=path, node_color="r", node_size=50)
nx.draw_networkx_edges(g, pos=node_positions, edgelist=path_edges, edge_color="g", width=4)
plt.axis("equal")
plt.show()
