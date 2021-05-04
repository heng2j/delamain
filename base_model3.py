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
import numpy as np
import cv2

from base.hud import HUD
from base.world import World
from base.manual_control import KeyboardControl
from lane_tracking.util.carla_util import CarlaSyncMode
from base.debug_cam import debug_view, save_img

from lane_tracking.cores.control.pure_pursuit import PurePursuitPlusPID
from lane_tracking.lane_track import lane_track_init, get_trajectory_from_lane_detector, get_speed, send_control
from lane_tracking.dgmd_track import image_pipeline

from gps_nav.nav_a2b import *
from gps_nav.geo_to_loc import geo_init, geo_to_location


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
        test_map = client.load_world('Town03')
        world = World(test_map, hud, args)
        controller = KeyboardControl(world, False)

        actor_list = []
        sensors = []

        # ==================================================================
        # TODO - features init/misc
        a_controller = PurePursuitPlusPID()
        cg, ld = lane_track_init()
        geo_model = geo_init()

        # TODO - add sensors
        blueprint_library = world.world.get_blueprint_library()

        # Camera RGB sensor
        bp_cam_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_cam_rgb.set_attribute('image_size_x', str(cg.image_width))
        bp_cam_rgb.set_attribute('image_size_y', str(cg.image_height))
        bp_cam_rgb.set_attribute('fov', str(cg.field_of_view_deg))

        # Semantic Segmentation camera
        bp_cam_seg = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp_cam_seg.set_attribute('image_size_x', str(cg.image_width))
        bp_cam_seg.set_attribute('image_size_y', str(cg.image_height))
        bp_cam_seg.set_attribute('fov', str(cg.field_of_view_deg))

        # GNSS sensor
        bp_gnss = blueprint_library.find('sensor.other.gnss')
        # TODO - mock destination
        destination = (-20.639827728271484, -142.1471405029297, 1.0)  # Fixed Location in Town03

        # Spawn Sensors
        transform = carla.Transform(carla.Location(x=0.7, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        cam_rgb = world.world.spawn_actor(bp_cam_rgb, transform, attach_to=world.player)
        print('created %s' % cam_rgb.type_id)
        cam_seg = world.world.spawn_actor(bp_cam_seg, transform, attach_to=world.player)
        print('created %s' % cam_seg.type_id)
        gnss_transform = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))
        gnss = world.world.spawn_actor(bp_gnss, gnss_transform, attach_to=world.player)
        print('created %s' % gnss.type_id)

        # Append actors / may not be necessary
        actor_list.append(cam_rgb)
        actor_list.append(cam_seg)
        actor_list.append(gnss)
        sensors.append(cam_rgb)
        sensors.append(cam_seg)
        sensors.append(gnss)
        # ==================================================================

        FPS = 30
        speed, traj, df_carla_path = 0, np.array([]), np.array([])
        time_cycle, cycles = 0.0, 30
        clock = pygame.time.Clock()
        # TODO - add sensor to SyncMode
        with CarlaSyncMode(world.world, cam_rgb, cam_seg, gnss, fps=FPS) as sync_mode:
            while True:
                clock.tick_busy_loop(FPS)
                time_cycle += clock.get_time()
                if controller.parse_events(client, world, clock):
                    return
                # Advance the simulation and wait for the data.
                tick_response = sync_mode.tick(timeout=2.0)
                # Data retrieval
                snapshot, image_rgb, image_seg, gnss_data = tick_response

                if time_cycle >= 1000.0/cycles:
                    time_cycle = 0.0

                    image_seg.convert(carla.ColorConverter.CityScapesPalette)
                    # ==================================================================
                    # TODO - run features
                    traj, lane_mask = get_trajectory_from_lane_detector(ld, image_seg) # stay in lane
                    current_loc = [gnss_data.latitude, gnss_data.longitude, gnss_data.altitude]
                    if world.gps_flag == True:
                        df_carla_path = process_nav_a2b(world.world, str(world.world.get_map().name), current_loc,
                                                        destination, dest_fixed=True, graph_vis=world.gps_vis, wp_vis=world.gps_vis)  # put dest_fixed=False if random location
                        world.gps_flag = False

                    ego_carla_loc = geo_to_location(gnss_data, geo_model)
                    print(ego_carla_loc)
                    # print(current_loc)

                    # dgmd_mask = image_pipeline(image_seg)
                    # save_img(image_seg)
                    # print(traj.shape, traj)

                    # ==================================================================
                    # Debug data
                    debug_view(image_rgb, image_seg, lane_mask)
                    # debug_view(image_rgb, image_seg)
                    # cv2.imshow("debug view", dgmd_mask)
                    # cv2.waitKey(1)
                if df_carla_path.any():
                    # next waypoint
                    # TODO - include in pipeline
                    row = df_carla_path.iloc[0]
                    target_x = row["x"]
                    target_y = row["y"]
                    target_z = row["z"]
                    target_transform = carla.Transform(carla.Location(x=target_x, y=target_y, z=target_z))
                    # print(target_transform.is_junction)

                # PID Control
                if world.autopilot_flag and traj.any():
                    speed = get_speed(world.player)
                    throttle, steer = a_controller.get_control(traj, speed, desired_speed=15, dt=1./FPS)
                    send_control(world.player, throttle, steer, 0)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
    finally:
        if (world and world.recording_enabled):
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
