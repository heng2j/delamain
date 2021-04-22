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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import pygame
import argparse
import random
import numpy as np

from base.hud import HUD
from base.world import World
from base.manual_control import KeyboardControl
from lane_tracking.util.carla_util import CarlaSyncMode
from base.debug_cam import debug_view

from lane_tracking.cores.control.pure_pursuit import PurePursuitPlusPID
from lane_tracking.lane_track import lane_track_init, get_trajectory_from_lane_detector, get_speed, send_control

from gps_nav.nav_a2b import *


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        test_map = client.load_world('Town02')
        world = World(test_map, hud, args)
        controller = KeyboardControl(world, False)

        actor_list = []
        sensors = []

        # ==================================================================
        # TODO - features init/misc
        a_controller = PurePursuitPlusPID()
        cg, ld = lane_track_init()

        # TODO - add sensors
        blueprint_library = world.world.get_blueprint_library()

        # Camera RGB sensor
        bp_cam_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_cam_rgb.set_attribute('image_size_x', str(cg.image_width))
        bp_cam_rgb.set_attribute('image_size_y', str(cg.image_height))
        bp_cam_rgb.set_attribute('fov', str(cg.field_of_view_deg))
        # bp_cam_rgb.set_attribute('sensor_tick', '0.0')

        # GNSS Sensor
        gnss_bp = blueprint_library().find('sensor.other.gnss')
        gnss_location = carla.Location(0, 0, 0)
        gnss_rotation = carla.Rotation(0, 0, 0)
        gnss_transform = carla.Transform(gnss_location, gnss_rotation)
        gnss_bp.set_attribute("sensor_tick", str(5.0))  # Wait time for sensor to update (1.0 = 1s)

        # Spawn Sensors
        transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        cam_rgb = world.world.spawn_actor(bp_cam_rgb, transform, attach_to=world.player)
        print('created %s' % cam_rgb.type_id)
        gnss = world.world.spawn_actor(gnss_bp, gnss_transform, attach_to=world.player)
        print('created %s' % gnss.type_id)

        # Append actors / may not be necessary
        actor_list.append(cam_rgb)
        sensors.append(cam_rgb)
        actor_list.append(gnss)
        sensors.append(gnss)
        # ==================================================================

        FPS = 30
        speed, traj = 0, np.array([])
        time_cycle, cycles = 0.0, 8
        clock = pygame.time.Clock()
        # TODO - add sensor to SyncMode
        with CarlaSyncMode(world.world, cam_rgb, fps=FPS) as sync_mode:
            while True:
                clock.tick_busy_loop(FPS)
                time_cycle += clock.get_time()
                if controller.parse_events(client, world, clock):
                    return
                # Advance the simulation and wait for the data.
                tick_response = sync_mode.tick(timeout=2.0)
                # Data retrieval
                snapshot, image_rgb = tick_response

                if time_cycle >= 1000.0/cycles:
                    time_cycle = 0.0

                    # ==================================================================
                    # TODO - run features
                    # traj, lane_mask = get_trajectory_from_lane_detector(ld, image_rgb)  # stay in lane

                    # TODO - NAV from A to B
                    # Get Town Map name
                    town = str(world.get_map().name)
                    # Activate GNSS Sensor
                    current_loc = gnss.listen(lambda event: gnss_live_location(event))
                    # Random Destination
                    # destination = random.choice(world.world.get_map().get_spawn_points())  # Random location
                    destination = (179.18255615234375, 215.42440795898438, 1.9653676748275757)  # Parking in Town02
                    # Get shortest path and visualize it
                    df_carla_path = process_nav_a2b(world, town, current_loc, destination)

                    # Gather and send waypoints one by one to vehicle
                    for index, row in df_carla_path.iterrows():
                        target_waypoint = row
                        target_x = row["x"]
                        target_y = row["y"]
                        target_z = row["z"]
                        ...

                    ##################################################################
                    # FROM CARLA DOC
                    # # Disable physics, in this example the vehicle is teleported.
                    # vehicle.set_simulate_physics(False)
                    # while True:
                    #     # Find next waypoint 2 meters ahead.
                    #     waypoint = random.choice(waypoint.next(2.0))
                    #     # Teleport the vehicle.
                    #     vehicle.set_transform(waypoint.transform)
                    ##################################################################

                    # ==================================================================
                    # Debug data
                    # debug_view(image_rgb, lane_mask)

                # PID Control
                if traj.any():
                    speed = get_speed(world.player)
                    throttle, steer = a_controller.get_control(traj, speed, desired_speed=15, dt=1./FPS)
                    send_control(world.player, throttle, steer, 0)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
    finally:
        if world and world.recording_enabled:
            client.stop_recorder()
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='Base Model Environment')
    args = argparser.parse_args()
    args.width, args.height = [1280, 720]

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
