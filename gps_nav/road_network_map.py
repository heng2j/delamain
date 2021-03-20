"""
ROAD NETWORK MAP

Script to create a visualization of the road network of the CARLA map.

The script is based on https://github.com/marcgpuig/carla_py_clients/blob/master/map_plot.py

Modified by DevGlitch
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
import matplotlib.pyplot as plt


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

        # Getting the topology of the map including pairs of waypoints.
        # The first element is the origin and the second one is the destination.
        topology = carla_map.get_topology()
        road_list = []

        for wp_pair in topology:
            current_wp = wp_pair[0]
            # Check if there is a road with no previous road, this can happen in OpenDrive. Then just continue.
            if current_wp is None:
                continue
            # First waypoint on the road that goes from wp_pair[0] to wp_pair[1].
            current_road_id = current_wp.road_id
            wps_in_single_road = [current_wp]
            # While current_wp has the same road_id (has not arrived to next road).
            while current_wp.road_id == current_road_id:
                # Check for next waypoints in approx distance.
                available_next_wps = current_wp.next(5.0)
                # If there is next waypoint/s?
                if available_next_wps:
                    # We must take the first ([0]) element because next(dist) can
                    # return multiple waypoints in intersections.
                    current_wp = available_next_wps[0]
                    wps_in_single_road.append(current_wp)
                else:  # If there is no more waypoints we can stop searching for more.
                    break
            road_list.append(wps_in_single_road)

        # Plot each road (on a different color by default)
        for road in road_list:
            plt.plot(
                [wp.transform.location.y for wp in road],
                [wp.transform.location.x for wp in road])

        # Display plot
        # plt.show()

        # Filename and directory using the name of the current carla map running
        directory = town + "_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = town + "_road_network_map.png"
        filepath = os.path.join(directory, filename)

        # Save Road Network Map
        if os.path.isfile(filepath):
            print(town, "road network map file already exist.")
        else:
            print("Saving", town, "road network map (png file).")
            plt.savefig(filepath)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        print('Done.')
