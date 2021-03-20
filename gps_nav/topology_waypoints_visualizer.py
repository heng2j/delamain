"""
TOPOLOGY WAYPOINTS VISUALIZER

Script to visualize the topology of the currently running CARLA map.
It will show the starting and ending waypoint for each tuple respectively in green and red .

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

        for t in topology_list:

            # To visualize each topology waypoint on the CARLA map
            # Starting waypoint (green)
            world.debug.draw_string(t[0].transform.location, 'O', draw_shadow=False,
                                    color=carla.Color(r=0, g=255, b=0), life_time=120.0,
                                    persistent_lines=True)
            # Ending waypoint (red)
            world.debug.draw_string(t[1].transform.location, 'X', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                    persistent_lines=True)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        print('Done.')
