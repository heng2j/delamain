import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import cv2
import numpy as np
import math
# import pygame
from gps_nav.nav_a2b import *


def main():
    actor_list = []

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Town05, Town07 has highway, parking, standard roads
        world = client.get_world()

        # blueprint library
        blueprint_library = world.get_blueprint_library()

        # Agent = Dodge Charge Police
        bp = blueprint_library.find('vehicle.dodge_charger.police')

        # GNSS Sensor
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_location = carla.Location(0, 0, 0)
        gnss_rotation = carla.Rotation(0, 0, 0)
        gnss_transform = carla.Transform(gnss_location, gnss_rotation)
        gnss_bp.set_attribute("sensor_tick", str(1.0))  # Wait time for sensor to update (1.0 = 1s)

        # Spawn Agent
        transform = carla.Transform(carla.Location(
            # x=121.61898803710938, y=187.5887451171875, z=1.0), carla.Rotation(yaw=180) # Location in Town02
            x=130.81553649902344, y=65.8092269897461, z=1.0), carla.Rotation(yaw=-0)  # Location in Town03

        )
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Spawn Sensors
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        actor_list.append(gnss)
        print('created %s' % gnss.type_id, "\n# # # # # # # # # # # # # # # # # # # # # #")

        # Get Town Map name
        town = str(world.get_map().name)

        # Activate GNSS Sensor and save current location in GNSS_DATA.parquet
        print("Starting GNSS sensor\n# # # # # # # # # # # # # # # # # # # # # #")
        gnss.listen(lambda event: gnss_live_location(event))
        time.sleep(2)
        print("Current location acquired.\n# # # # # # # # # # # # # # # # # # # # # #")

        # Current Location - Loads it from GNSS_DATA.parquet
        gnss_data = pd.read_parquet("GNSS_DATA.parquet")
        current_loc = (gnss_data.loc[0, 'lat'], gnss_data.loc[0, 'lon'], gnss_data.loc[0, 'alt'])

        # Fixed Destination (CARLA Location of type x, y, z)
        # destination = (127.02777862548828, 306.4728088378906, 1.0)  # Fixed Location in Town02
        destination = (-20.639827728271484, -142.1471405029297, 1.0)  # Fixed Location in Town03

        # Random Destination (CARLA Location)
        # If choosing this option instead of the fixed destination make sure to
        # change dest_fixed=False in the code below
        # destination = random.choice(world.world.get_map().get_spawn_points())

        # Get shortest path and visualize it
        df_carla_path = process_nav_a2b(world, town, current_loc, destination, dest_fixed=True)  # put dest_fixed=False if random location

        # Vehicle autopilot
        vehicle.set_autopilot(False)  # Not activated if False (Needs to be False for teleportation)

        # Vehicle physics
        vehicle.set_simulate_physics(False)  # For teleportation put False

        # Gather and send waypoints one by one to vehicle
        print("Driving to destination...")
        for index, row in df_carla_path.iterrows():
            target_x = row["x"]
            target_y = row["y"]
            target_z = row["z"]

            # Find next waypoint 2 meters ahead.
            target_waypoint = carla.Transform(carla.Location(x=target_x, y=target_y, z=target_z+1))  # +1 in z for teleport

            # Teleport the vehicle.
            vehicle.set_transform(target_waypoint)

            time.sleep(1)

        print("# # # # # # # # # # # # # # # # # # # # # #\n"
              "You have arrived at your destination.\n")

    finally:
        print('\nDestroying actors.')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        if os.path.exists("GNSS_DATA.parquet"):
            print('Destroying GNSS_DATA.parquet.')
            os.remove("GNSS_DATA.parquet")
        print('Done.')


if __name__ == '__main__':
    main()
