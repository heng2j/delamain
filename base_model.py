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


def main():
    actor_list = []

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Town05, Town07 has highway, parking, standard roads
        world = client.get_world()

        # blueprint library
        blueprint_library = world.get_blueprint_library()

        # Agent = Tesla model 3
        bp = blueprint_library.find('vehicle.tesla.model3')

        # Spawn Agent
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        # vehicle.set_autopilot(True)

        # Agent time to live
        time.sleep(30)

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    main()
