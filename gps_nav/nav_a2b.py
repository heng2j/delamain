import carla
import os
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy import spatial
import matplotlib.pyplot as plt


def gnss_live_location(event):
    """Get, print, and store the GNSS Measurements.
    :return: Float values of Latitude, and Longitude, and Altitude
    """

    latitude = event.latitude  # degrees
    longitude = event.longitude  # degrees
    altitude = event.altitude  # meters

    # print("GNSS measure:\n" + str(event) + '\n')  # For full details of measurements incl. frame and timestamp
    # print("latitude:", str(latitude))
    # print("longitude:", str(longitude))
    # print("altitude:", str(altitude), "\n")

    df = pd.DataFrame(columns=["lat", "lon", "alt"])
    data = {'lat': latitude, 'lon': longitude, 'alt': altitude}
    df = df.append(data, ignore_index=True)

    # saving the dataframe
    df.to_parquet('GNSS_DATA.parquet')


def find_closest_node(node_list, node_array, location):
    """
    Find the closest node to a geolocation
    Returns the Node ID, Latitude, Longitude, and Altitude.
    !! For use in shortest_path !!
    """
    distance, index = spatial.KDTree(node_array).query(location)
    return node_list.iloc[index]


def shortest_path(town_map, start_location, end_location, vis=False):
    """
    This script creates a networkx directed graph using the topology data previously created (geolocations).
    Find the closest node of the starting and ending locations.
    Then, calculates the shortest path between the two nodes.
    Finally show the shortest path on the graph as well as return it as dataframe.

    @return: dataframe with geolocations of each waypoint
    """

    # Directory name in which the specific town data is
    directory = town_map + "_data"

    # Edge list file name
    edge_filename = town_map + "_topology_edge_list.parquet"
    edge_filepath = os.path.join("gps_nav", directory, edge_filename)

    # Grab edge list data
    edgelist = pd.read_parquet(edge_filepath)

    # Node list file name
    node_filename = town_map + "_topology_node_list.parquet"
    node_filepath = os.path.join("gps_nav", directory, node_filename)

    # Grab node list data hosted on Gist
    nodelist = pd.read_parquet(node_filepath)

    # Create empty directed graph
    g = nx.DiGraph()

    # Add edges and edge attributes
    for i, elrow in edgelist.iterrows():
        g.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())

    # Add node attributes
    for i, nlrow in nodelist.iterrows():
        g.nodes[nlrow["id"]].update(nlrow[1:].to_dict())

    # Subset dataframe of node list incl. only lat and lon coordinates
    node_geo = nodelist[["lat", "lon"]]

    # Create array for use with spicy
    node_geo_array = np.array(node_geo)

    # Get closest node to starting location (1st commented out option to use if
    start_location_closest_node = find_closest_node(
        node_list=nodelist, node_array=node_geo_array, location=start_location[0:2]
    )

    # Get closest node to ending location / destination
    end_location_closest_node = find_closest_node(
        node_list=nodelist, node_array=node_geo_array, location=(end_location.latitude, end_location.longitude)
    )

    # Compute shortest path between the two nodes closest to start and end locations
    # Return a list with node IDs with the first value being the starting node and the last value the ending node.
    shortest_path = nx.shortest_path(
        g,
        source=start_location_closest_node[0],
        target=end_location_closest_node[0],
        weight="distance",
        method="dijkstra"
    )

    # Get lat, lon, and alt attributes of each nodes
    shortest_path_geo = pd.DataFrame(columns=["id", "lat", "lon", "alt"])

    for i in shortest_path:
        node_attributes = nodelist.loc[nodelist["id"] == i]
        shortest_path_geo = shortest_path_geo.append(node_attributes)

    # !! QUICK FIX !! - Removing the last row prior adding destination in order to avoid the car too go too far
    shortest_path_geo = shortest_path_geo[:-1]

    # Append destination to dataframe (ID 999 is only used for destination)
    destination_attributes = {"id": 999, "lat": end_location.latitude, "lon": end_location.longitude, "alt": end_location.altitude}
    shortest_path_geo = shortest_path_geo.append(destination_attributes, ignore_index=True)

    # Define node positions data structure (dict) for plotting
    node_positions = {
        node[0]: (node[1]["lat"], -node[1]["lon"]) for node in g.nodes(data=True)
    }

    if vis:
        # Create a directed graph and overlay the shortest path on it
        # This is useful for a demo in order to show GPS navigation output on 2D map
        plt.figure(3, figsize=(8, 6))
        nx.draw_networkx_nodes(g, pos=node_positions, node_size=20, node_color="red")
        nx.draw_networkx_edges(g, pos=node_positions, edge_color="blue", arrows=False)
        path = shortest_path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(
            g, pos=node_positions, nodelist=path, node_color="r", node_size=50
        )
        nx.draw_networkx_edges(
            g, pos=node_positions, edgelist=path_edges, edge_color="g", width=4
        )
        print("Close plot to continue...")
        plt.show()
        print("# # # # # # # # # # # # # # # # # # # # # #")

    return shortest_path_geo


def from_gps_to_xyz(latitude: float, longitude: float, altitude: float):
    """Get carla location x y z coordinates from GPS (latitude, longitude, altitude)."""

    # Equatorial mean radius of Earth in meters
    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    earth_radius = 6378137.0

    # Hardcoded can only work for town 01 to 07
    lat_ref = 0.0
    lon_ref = 0.0

    scale = math.cos(lat_ref * math.pi / 180.0)
    base_x = scale * lon_ref * math.pi * earth_radius / 180.0
    base_y = scale * earth_radius * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))

    x = scale * (longitude - base_x) * math.pi * earth_radius / 180.0
    y = (scale * earth_radius * math.log(math.tan((90.0 + latitude) * math.pi / 360.0)) - base_y) * -1
    z = altitude

    # Like carla.Location
    carla_location = (x, y, z)

    return carla_location


def carla_geo_clean(carla_geo_location):
    """Retrieve the id, lat, lon, alt from a geo transform Carla.location"""
    lat = [carla_geo_location.latitude]
    lon = [carla_geo_location.longitude]
    alt = [carla_geo_location.altitude]

    geo_location = (lat, lon, alt)

    return geo_location


def get_carla_path(shortest_path_geo):
    """
    Generate the carla path by converting the shortest_path_geo dataframe
    @param shortest_path_geo: dataframe from shortest_path_geo function
    @return: dataframe of carla locations path
    """
    # Initialise lists
    id = []
    x = []
    y = []
    z = []

    # Convert each single geolocation to carla location
    for index, row in shortest_path_geo.iterrows():

        id += [row["id"]]

        lat = row["lat"]
        lon = row["lon"]
        alt = row["alt"]
        path = from_gps_to_xyz(lat, lon, alt)

        x += [path[0]]
        y += [path[1]]
        z += [path[2]]

    # Transform into dataframe
    carla_path = pd.DataFrame(
            list(zip(id, x, y, z)),
            columns=["id", "x", "y", "z"])

    return carla_path


def shortest_path_visualizer(world, path):

    for index, row in path.iterrows():

        lat = row["lat"]
        lon = row["lon"]
        alt = row["alt"]

        # Converting geolocation coordinates to carla x y z coordinates (in meters) called a,b,c due to a bug
        a, b, c = from_gps_to_xyz(lat, lon, alt)

        # To visualize each waypoint on the CARLA map

        # Starting waypoint (green)
        if index == 0:
            world.debug.draw_string(
                carla.Location(a, b, c + 1),
                "START",
                draw_shadow=False,
                color=carla.Color(r=255, g=64, b=0),
                life_time=30.0,
                persistent_lines=False,
            )
            continue

        # Ending waypoint (red)
        if index == path.last_valid_index():
            world.debug.draw_string(
                carla.Location(a, b, c + 1),
                "END",
                draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0),
                life_time=30.0,
                persistent_lines=False,
            )

        # Waypoints between start and finish (blue)
        else:
            world.debug.draw_string(
                carla.Location(a, b, c + 1),
                "X",
                draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255),
                life_time=30.0,
                persistent_lines=False,
            )


def process_nav_a2b(world, town, current_loc, destination, dest_fixed=True, graph_vis=True, wp_vis=True):

    print("Launching GPS Navigation...\n# # # # # # # # # # # # # # # # # # # # # #")

    # Cleaning up the destination info
    # For fixed/hardcoded destination (see base_model)
    if dest_fixed:
        destination_geo = world.get_map().transform_to_geolocation(carla.Location(x=destination[0], y=destination[1], z=destination[2]))
    # For random CARLA location (see base_model)
    else:
        destination_geo = carla_geo_clean(world.get_map().transform_to_geolocation(destination))

    print("Your have selected as destination:", destination_geo, "\n# # # # # # # # # # # # # # # # # # # # # #")

    # Get shortest path + graph visual
    df_geo_path = shortest_path(town_map=town, start_location=current_loc, end_location=destination_geo, vis=graph_vis)
    print("We have found the shortest path thanks to the Dijkstra's algorithm.\n# # # # # # # # # # # # # # # # # # # # # #")

    # Visualize path on Carla map
    if wp_vis:
        print("Visualizing the shortest path on CARLA map...\n# # # # # # # # # # # # # # # # # # # # # #")
        shortest_path_visualizer(world, df_geo_path)

    # Convert geo path to carla location (x,y,z) path
    df_carla_path = get_carla_path(df_geo_path)

    return df_carla_path
