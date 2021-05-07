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
from intersection.global_route_planner_dao import GlobalRoutePlannerDAO
from intersection.global_route_planner import GlobalRoutePlanner

from gps_nav.nav_a2b import *

from car_chasing.car_chasing_agent import chasing_car_init, agent_init
from car_chasing.car_chasing_controller import ChaseControl


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(100.0)

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
        b_controller = setup_gps_pid(world.player)
        routeplanner = GlobalRoutePlanner(GlobalRoutePlannerDAO(world.world.get_map(), 2))
        routeplanner.setup()
        chaseControl = ChaseControl()
        # trailing_vehicle = chasing_car_init(world=world, position=None, y_offset=y_offset)
        # actor_list.append(trailing_vehicle)

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
        destination = (79.0, -216, 0.0)  # Fixed Location in Town03


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

        frame = 0
        FPS = 30
        speed, traj = 0, np.array([])
        df_carla_path, wp_counter, final_wp, wp_distance, route_distance = pd.DataFrame(), 0, False, 0, 0
        time_cycle, cycles, wp_cycle = 0.0, 30, 0.0
        route = False
        pid_type, visual_nav = "N/A", True
        target_spawn = False
        clock = pygame.time.Clock()
        # TODO - add sensor to SyncMode
        with CarlaSyncMode(world.world, cam_rgb, cam_seg, gnss, fps=FPS) as sync_mode:
            while True:
                clock.tick_busy_loop(FPS)
                time_cycle += clock.get_time()
                wp_cycle += clock.get_time()
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
                    if world.gps_flag:
                        df_carla_path = process_nav_a2b(world.world, str(world.world.get_map().name), current_loc,
                                                        destination, dest_fixed=True, graph_vis=world.gps_vis, wp_vis=world.gps_vis)  # put dest_fixed=False if random location
                        world.gps_flag = False
                        wp_counter = 1
                        route = None

                    if not df_carla_path.empty and not final_wp:
                        # next waypoint
                        row = df_carla_path.iloc[wp_counter]
                        next_carla_loc = carla.Location(x=row["x"], y=row["y"], z=row["z"])

                        # arrive @ next waypoint
                        wp_distance = next_carla_loc.distance(ego_carla_loc)
                        if wp_distance <= 5.0:
                            wp_counter += 1
                            route = None
                            # last waypoint
                            if wp_counter == len(df_carla_path)-1:
                                visual_nav = False
                                final_wp = True
                                route = routeplanner.trace_route(ego_carla_loc, next_carla_loc)
                                route_counter = 1


                    ego_carla_loc = world.player.get_location()

                    # dgmd_mask = image_pipeline(image_seg)

                    # manual data collection
                    if world.save_img:
                        save_img(image_rgb, path='intersection/data/image_%s.png')
                        world.save_img = False

                    # ==================================================================
                    # Debug data
                    debug_view(image_rgb, image_seg, lane_mask,
                               text=[not visual_nav, pid_type, round(wp_distance, 2), wp_counter, world.car_chase])

                # Car chasing
                if world.car_chase:
                    # Spawn target vehicle
                    if not target_spawn:
                        target_bp = blueprint_library.find('vehicle.tesla.model3')
                        target_transform = carla.Transform(carla.Location(x=ego_carla_loc.x+10,y=ego_carla_loc.y,z=1), carla.Rotation(yaw=-180))
                        target_vehicle = world.world.spawn_actor(target_bp, target_transform)
                        target_vehicle.set_autopilot(True)
                        target_spawn = True
                    # trailing_steer, trailing_throttle, real_dist = chaseControl.behaviour_planner(
                    #     leading_vehicle=target_vehicle,
                    #     trailing_vehicle=world.player,
                    #     trailing_image_seg=image_seg,
                    #     trail_cam_rgb=image_rgb,
                    #     frame=frame)
                    # send_control(target_vehicle, trailing_throttle, trailing_steer, 0)
                    # frame += 1

                # PID Controls
                if world.autopilot_flag and not world.car_chase:
                    if wp_cycle >= 1000.0 and not final_wp:
                        wp_cycle = 0.0
                        visual_nav = not world.world.get_map().get_waypoint(ego_carla_loc).next(2)[0].is_junction
                    if visual_nav and traj.any():
                        pid_type = "visual"
                        speed = get_speed(world.player)
                        throttle, steer = a_controller.get_control(traj, speed, desired_speed=15, dt=1. / FPS)
                        send_control(world.player, throttle, steer, 0)
                        route = None
                    else:
                        pid_type = "gps"
                        if not route:
                            route = routeplanner.trace_route(ego_carla_loc, next_carla_loc)
                            route_counter = 0
                        if route_counter == len(route)-1:
                            route = None
                            continue
                        wp = route[-1][0]
                        if route_counter < len(route)-4:
                            world.world.debug.draw_point(route[route_counter+4][0].transform.location,
                                                         color=carla.Color(r=0, g=255, b=0), size=0.1,
                                                         life_time=120.0)
                        route_distance = wp.transform.location.distance(ego_carla_loc)
                        if route_distance <= 2:
                            route_counter += 1
                        elif final_wp and route_distance <= 5:
                            gps_speed = 0
                            world.autopilot_flag = False
                        else:
                            gps_speed = 20
                        control = gps_pid(wp, gps_speed, b_controller)
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
