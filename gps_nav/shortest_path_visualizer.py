"""
SHORTEST PATH VISUALIZER

......

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

        # Path - Hardcoded for development right now
        path = [9.806409144428777e+18,
                1.7966488129604989e+19,
                7.991834403214065e+18,
                8.63726057961196e+18,
                2.393595647993709e+18,
                1.1830260409854511e+19]

        for i in path:

            # Need to draw on the carla environment every single waypoint of the path
            # Maybe green for start, red for end, and orange in between?
            ...

            # To visualize each waypoint on the CARLA map
            # Starting waypoint (green)
            world.debug.draw_string(i[0].transform.location, 'O', draw_shadow=False,
                                    color=carla.Color(r=0, g=255, b=0), life_time=120.0,
                                    persistent_lines=True)
            # Ending waypoint (red)
            world.debug.draw_string(i[1].transform.location, 'X', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                    persistent_lines=True)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        print('Done.')
