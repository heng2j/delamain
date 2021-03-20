"""
TOPOLOGY GEOLOCATIONS TO CSV

Script to acquire the topology of the currently running CARLA map.
Then transforming every single waypoint location to geolocations coordinates (latitude, longitude, altitude).
Also calculates the distance between waypoints of each tuple.
Finally getting this data into a DataFrame and saving it locally into a CSV file.

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

        # Topology waypoints (tuple type)
        topology_list = carla_map.get_topology()

        # Initializing lists
        lat_start = []
        lon_start = []
        alt_start = []
        lat_end = []
        lon_end = []
        alt_end = []
        road_id_start = []
        road_id_end = []
        distance = []

        for t in topology_list:

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

            # Get Road ID of each tuple
            road_id_start += [t[0].road_id]
            road_id_end += [t[1].road_id]

            # Calculate the distance in kilometer between each waypoint of each tuple
            # Great-circle distance formula: https://en.wikipedia.org/wiki/Great-circle_distance
            lat_0 = np.radians(waypoint_geo_0.latitude)
            lon_0 = np.radians(waypoint_geo_0.longitude)
            lat_1 = np.radians(waypoint_geo_1.latitude)
            lon_1 = np.radians(waypoint_geo_1.longitude)
            diff_lon = lon_0 - lon_1
            earth_radius = 6371  # radius of the earth in km ### THIS CAN BE CHANGE TO METERS IF NEEDED ###

            y = np.sqrt((np.cos(lat_1) * np.sin(diff_lon)) ** 2 +
                        (np.cos(lat_0) * np.sin(lat_1) - np.sin(lat_0) *
                         np.cos(lat_1) * np.cos(diff_lon)) ** 2)
            x = np.sin(lat_0) * np.sin(lat_1) + \
                np.cos(lat_0) * np.cos(lat_1) * np.cos(diff_lon)
            c = np.arctan2(y, x)

            distance += [earth_radius * c]

        # Create DataFrame
        df = pd.DataFrame(
            list(zip(road_id_start, lat_start, lon_start, alt_start, road_id_end, lat_end, lon_end, alt_end, distance)),
            columns=["road_id_start", "lat_start", "lon_start", "alt_start", "road_id_end", "lat_end", "lon_end", "alt_end", "distance"]
        )

        # Filename using the name of the current carla map running
        filename = str(world.get_map().name) + "_topology.csv"

        # Save DataFrame to local CSV file
        df.to_csv(filename)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        print('Done.')
