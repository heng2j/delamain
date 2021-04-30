"""
Spectator Location

This script gives you the exact carla location in x,y,z format of the spectator position.
Can be used to retrieve specific locations like parking, a desired destination etc.

Beginning part of the code is based on: https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial

Created by DevGlitch
"""

import glob
import os
import sys
import time


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

        while True:
            spec_transform = world.get_spectator().get_transform()
            xyz_location = "(x,y,z) = ({},{},{})".format(spec_transform.location.x, spec_transform.location.y, spec_transform.location.z)
            print(xyz_location)
            time.sleep(5)

    finally:
        pass


if __name__ == '__main__':
    main()
