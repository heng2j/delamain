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

import pygame
import argparse

from base.hud import HUD
from base.world import World
from base.manual_control import KeyboardControl
from lane_tracking.util.carla_util import CarlaSyncMode
from base.debug_cam import debug_view, save_img

from lane_tracking.cores.control.pure_pursuit import PurePursuitPlusPID
from lane_tracking.lane_track import lane_track_init, get_trajectory_from_lane_detector, get_speed, send_control
from intersection.intersection import setup_gps_pid, gps_pid

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
        b_controller = setup_gps_pid(world.player)

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
        # destination = (-20.5, -142.0, 1.0)  # Fixed Location in Town03
        destination = (32.0, 137.7, 1.0)  # Fixed Location in Town03


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
        speed, traj = 0, np.array([])
        df_carla_path, wp_counter, nextwp, warning_timer = pd.DataFrame(), 0, None, 0.0
        time_cycle, cycles = 0.0, 30
        pid_type, junction = "N/A", False
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
                    traj, lane_mask, poly_warning = get_trajectory_from_lane_detector(ld, image_seg) # stay in lane
                    current_loc = [gnss_data.latitude, gnss_data.longitude, gnss_data.altitude]
                    if world.gps_flag == True:
                        df_carla_path = process_nav_a2b(world.world, str(world.world.get_map().name), current_loc,
                                                        destination, dest_fixed=True, graph_vis=world.gps_vis, wp_vis=world.gps_vis)  # put dest_fixed=False if random location
                        world.gps_flag = False
                        wp_counter = 1

                    if not df_carla_path.empty:
                        # next waypoint
                        # TODO - include in pipeline
                        row = df_carla_path.iloc[wp_counter]
                        next_carla_loc = carla.Location(x=row["x"], y=row["y"], z=row["z"])
                        if nextwp == None:
                            nextwp = world.world.get_map().get_waypoint(next_carla_loc)

                        distance = next_carla_loc.distance(ego_carla_loc)
                        print("distance:", distance)
                        if distance <= 1.0:
                            wp_counter += 1
                            nextwp = None
                            if wp_counter == len(df_carla_path):
                                wp_counter = 0
                                world.autopilot_flag = False

                    ego_carla_loc = geo_to_location(gnss_data, geo_model)
                    if poly_warning and warning_timer >= 1000.0:
                        junction = True
                    elif poly_warning:
                        warning_timer += clock.get_time()
                    else:
                        junction = False
                        warning_timer = 0.0

                    # dgmd_mask = image_pipeline(image_seg)

                    # manual data collection
                    if world.save_img:
                        save_img(image_rgb, path='intersection/data/image_%s.png')
                        world.save_img = False

                    # ==================================================================
                    # Debug data
                    debug_view(image_rgb, image_seg, lane_mask, text=[junction, pid_type])
                print(warning_timer)
                # PID Controls
                # visual based pid - lane
                if world.autopilot_flag and traj.any() and not junction:
                    pid_type = "visual"
                    speed = get_speed(world.player)
                    throttle, steer = a_controller.get_control(traj, speed, desired_speed=15, dt=1./FPS)
                    send_control(world.player, throttle, steer, 0)
                # gps based pid - intersection
                elif world.autopilot_flag and junction and wp_counter > 0 and nextwp is not None:
                    pid_type = "gps"
                    control = gps_pid(nextwp, 5, b_controller)
                    world.player.apply_control(control)

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
