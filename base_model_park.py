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
from self_parking.maneuvres import *


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
        # transform = carla.Transform(carla.Location(x=36, y=193.4, z=0.05), carla.Rotation(yaw=180))  # Parallel
        transform = carla.Transform(carla.Location(x=223.5, y=54, z=0.05), carla.Rotation(yaw=-90))  # Perpendicular
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)

        # Self-Parking Maneuvres
        # Please unsure to run only one of these below functions +
        # Ensure to update the spawn location of the vehicle

        # Parallel parking
        # parallel_parking(world, actor_list, vehicle)

        # Perpendicular parking
        perpendicular_parking(world, actor_list, vehicle)

        print("Vehicle successfully parked.")

    finally:
        print('\nDestroying actors.')
        for actor in actor_list:
            actor.destroy()
        print('Done.')


if __name__ == '__main__':
    main()
