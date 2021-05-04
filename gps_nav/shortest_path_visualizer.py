"""
SHORTEST PATH VISUALIZER

......

Created by DevGlitch
"""

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla
import argparse
import pandas as pd
from transform_geo_to_carla_xyz import from_gps_to_xyz


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        world = client.get_world()

        # carla_map = world.get_map()

        # Path - CSV file only for development - It will use the DF directly
        path = pd.read_csv("test_path_Town02.csv")
        # Dropping the first column as pd.to_csv created a column for the index
        path.drop(path.columns[0], axis=1, inplace=True)
        # For debug printing the dataframe
        # print(path, "\n\n\n")

        for index, row in path.iterrows():

            # id = row["id"]
            lat = row["lat"]
            lon = row["lon"]
            alt = row["alt"]

            # For debug printing each row
            # print("index:", index, "\nid=", id, "\nlat=", lat, "\nlon=", lon, "\nalt=", alt, "\n")

            # Converting geolocation coordinates to carla x y z coordinates (in meters)
            a, b, c = from_gps_to_xyz(lat, lon, alt)

            # print("\na=", a, "\nb=", b, "\nc=", c)

            # For debug
            # print("id=", id, "\nx=", x, "\ny=", y, "\nz=", z, "\n")

            # Need to draw on the carla environment every single waypoint of the path
            # Maybe green for start, red for end, and orange in between?

            # To visualize each waypoint on the CARLA map
            # Starting waypoint (green)
            if index == 0:
                world.debug.draw_string(
                    carla.Location(a, b, c + 1),
                    "START",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=64, b=0),
                    life_time=10.0,
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
                    life_time=10.0,
                    persistent_lines=False,
                )

            # Waypoints between start and finish (blue)
            else:
                world.debug.draw_string(
                    carla.Location(a, b, c + 1),
                    "X",
                    draw_shadow=False,
                    color=carla.Color(r=0, g=0, b=255),
                    life_time=10.0,
                    persistent_lines=False,
                )

    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Done.")
