"""
TOPOLOGY EDGE AND NODE

Script to acquire the topology of the currently running CARLA map.
Then transforming every single waypoint carla location to geolocations coordinates (latitude, longitude, altitude).
Also calculates the distance between waypoints of each tuple.
Finally getting this data into DataFrame and saving it locally into two separate parquet files.

Created by DevGlitch
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import pandas as pd
import pyarrow
import numpy as np


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        world = client.get_world()
        carla_map = world.get_map()

        town = str(world.get_map().name)

        # Topology waypoints (tuple type)
        topology_list = carla_map.get_topology()

        # # Initializing lists
        id_start = []
        id_end = []
        lat_start = []
        lon_start = []
        alt_start = []
        lat_end = []
        lon_end = []
        alt_end = []
        distance = []

        for t in topology_list:

            # Get Waypoint ID
            id_start += [t[0].id]
            id_end += [t[1].id]

            # Convert waypoints carla locations to geolocations
            # Starting waypoints
            waypoint_geo_0 = carla_map.transform_to_geolocation(t[0].transform.location)
            lat_start += [waypoint_geo_0.latitude]
            lon_start += [waypoint_geo_0.longitude]
            alt_start += [waypoint_geo_0.altitude]
            # Ending waypoints
            waypoint_geo_1 = carla_map.transform_to_geolocation(t[1].transform.location)
            lat_end += [waypoint_geo_1.latitude]
            lon_end += [waypoint_geo_1.longitude]
            alt_end += [waypoint_geo_1.altitude]

            # Calculate the distance in kilometer between each waypoint of each tuple
            # Using the Great-circle distance formula: https://en.wikipedia.org/wiki/Great-circle_distance
            lat_0 = np.radians(waypoint_geo_0.latitude)
            lon_0 = np.radians(waypoint_geo_0.longitude)
            lat_1 = np.radians(waypoint_geo_1.latitude)
            lon_1 = np.radians(waypoint_geo_1.longitude)
            diff_lon = lon_0 - lon_1
            earth_radius = 6371  # radius of the earth in km

            y = np.sqrt((np.cos(lat_1) * np.sin(diff_lon)) ** 2 +
                        (np.cos(lat_0) * np.sin(lat_1) - np.sin(lat_0) *
                         np.cos(lat_1) * np.cos(diff_lon)) ** 2)
            x = np.sin(lat_0) * np.sin(lat_1) + \
                np.cos(lat_0) * np.cos(lat_1) * np.cos(diff_lon)
            c = np.arctan2(y, x)

            distance += [earth_radius * c]

        ######################################################################################
        # ############################CREATING NODE LIST#################################### #
        ######################################################################################

        # Create DataFrame for start waypoint
        df_start = pd.DataFrame(
            list(zip(id_start, lat_start, lon_start, alt_start)),
            columns=["id", "lat", "lon", "alt"]
        )

        # Create DataFrame for end waypoint
        df_end = pd.DataFrame(
            list(zip(id_end, lat_end, lon_end, alt_end)),
            columns=["id", "lat", "lon", "alt"]
        )

        # Merge both DataFrame to make the Node List Dataframe
        df = pd.concat([df_start, df_end])

        # Remove duplicate waypoints
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        # Filename and directory using the name of the current carla map running
        directory = town + "_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = town + "_topology_node_list.parquet"
        filepath = os.path.join(directory, filename)

        # Save DataFrame to local parquet file
        if os.path.isfile(filepath):
            print(town, "node list file already exist.")
        else:
            print("Saving", town, "node list dataframe to parquet.")
            df.to_parquet(filepath)

        ######################################################################################
        # ############################CREATING EDGE LIST#################################### #
        ######################################################################################

        # Create Edge List DataFrame
        df = pd.DataFrame(
            list(zip(id_start, id_end, distance)),
            columns=["id_start", "id_end", "distance"]
        )

        # Filename and directory using the name of the current carla map running
        directory = town + "_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = town + "_topology_edge_list.parquet"
        filepath = os.path.join(directory, filename)

        # Save DataFrame to local parquet file
        if os.path.isfile(filepath):
            print(town, "edge list file already exist.")
        else:
            print("Saving", town, "edge list dataframe to parquet.")
            df.to_parquet(filepath)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        print('Done.')
