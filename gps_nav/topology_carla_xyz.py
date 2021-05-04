"""
TOPOLOGY CARLA XYZ

Script to acquire the topology of the currently running CARLA map.
Printing the carla location (x,y,z) waypoints

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

        id0 = []
        x0 = []
        y0 = []
        z0 = []
        id1 = []
        x1 = []
        y1 = []
        z1 = []

        for t in topology_list:
            # Waypoints

            id0 += [t[0].id]
            x0 += [t[0].transform.location.x]
            y0 += [t[0].transform.location.y]
            z0 += [t[0].transform.location.z]

            id1 += [t[1].id]
            x0 += [t[1].transform.location.x]
            y0 += [t[1].transform.location.y]
            z0 += [t[1].transform.location.z]

        df0 = pd.DataFrame(
            list(zip(id0, x0, y0, z0)),
            columns=["id", "x", "y", "z"]
        )

        df1 = pd.DataFrame(
            list(zip(id1, x1, y1, z1)),
            columns=["id", "x", "y", "z"]
        )

        # Merge both DataFrame
        df = pd.concat([df0, df1])

        # Remove duplicate waypoints
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        # Filename and directory using the name of the current carla map running
        directory = town + "_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = town + "_topology_xyz_list.csv"
        filepath = os.path.join(directory, filename)

        # Save DataFrame to local parquet file
        if os.path.isfile(filepath):
            print(town, "xyz list file already exist.")
        else:
            print("Saving", town, "xyz list dataframe to csv.")
            df.to_csv(filepath)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        print('Done.')
