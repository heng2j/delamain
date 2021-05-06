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


# Car chasing imports
# New imports
from car_chasing.DrivingControl import DrivingControl
from car_chasing.DrivingControlAdvanced import DrivingControlAdvanced
from car_chasing.CarDetector import CarDetector
from car_chasing.SemanticSegmentation import SemanticSegmentation


# ==============================================================================
# -- Car Chasing Objects ---------------------------------------------------------------
# ==============================================================================

carDetector = CarDetector()
drivingControl = DrivingControl(optimalDistance=optimalDistance)
drivingControlAdvanced = DrivingControlAdvanced(optimalDistance=optimalDistance)
semantic = SemanticSegmentation()

# ==============================================================================
# -- Car Chasing Configuration Variables ---------------------------------------------------------------
# ==============================================================================

optimalDistance = 8
nOfFramesToSkip = 0
extrapolate = True
LP_FREQUENCY_DIVISOR = 2
y_offset = 10

# For object avoidance
obsticle_vehicleSpawned = False

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

        # Spawn Sensors
        transform = carla.Transform(carla.Location(x=0.7, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        cam_rgb = world.world.spawn_actor(bp_cam_rgb, transform, attach_to=world.player)
        print('created %s' % cam_rgb.type_id)
        cam_seg = world.world.spawn_actor(bp_cam_seg, transform, attach_to=world.player)
        print('created %s' % cam_seg.type_id)

        # Append actors / may not be necessary
        actor_list.append(cam_rgb)
        actor_list.append(cam_seg)
        sensors.append(cam_rgb)
        sensors.append(cam_seg)
        # ==================================================================


        # ======================= Add Trailing Car =========================

        # Set Trailing vehicle
        m = world.world.get_map()

        trailing_vehicle_bp = blueprint_library.filter('vehicle.dodge_charger.police')[0]
        trailing_vehicle = world.world.spawn_actor(
            trailing_vehicle_bp,
            m.get_spawn_points()[90])
        
        actor_list.append(trailing_vehicle)

        # Set position for trailing vehicle
        lead_vehicle_transfrom = world.player.get_transform()


        # TODO - Set trailing offset  
        start_pose = carla.Transform()
        start_pose.rotation = lead_vehicle_transfrom.rotation
        start_pose.location.x = lead_vehicle_transfrom.location.x 
        start_pose.location.y =  lead_vehicle_transfrom.location.y + y_offset 
        start_pose.location.z =  lead_vehicle_transfrom.location.z

        trailing_vehicle.set_transform(start_pose)
        trailing_vehicle.set_simulate_physics(True)


        # Adding RGB camera
        trail_cam_rgb_blueprint = world.world.get_blueprint_library().find('sensor.camera.rgb')
        trail_cam_rgb_blueprint.set_attribute('fov', '90')
        trail_cam_rgb = world.world.spawn_actor(
           trail_cam_rgb_blueprint,
            carla.Transform(carla.Location(x=1.5, z=1.4,y=0.3), carla.Rotation(pitch=0)),
            attach_to=trailing_vehicle)
        actor_list.append(trail_cam_rgb)
        sensors.append(trail_cam_rgb)

        # Adding Segmentation camera
        trail_cam_seg = world.world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=1.5, z=1.4,y=0), carla.Rotation(pitch=0)), #5,3,0 # -0.3
            attach_to=trailing_vehicle)
        actor_list.append(trail_cam_seg)
        sensors.append(trail_cam_seg)

        
        # ==================================================================
        frame = 0 
        FPS = 30
        speed, traj = 0, np.array([])
        time_cycle, cycles = 0.0, 30
        clock = pygame.time.Clock()
        # TODO - add sensor to SyncMode
        with CarlaSyncMode(world.world, *sensors, fps=FPS) as sync_mode:
            while True:
                clock.tick_busy_loop(FPS)
                time_cycle += clock.get_time()
                if controller.parse_events(client, world, clock):
                    return
                # Advance the simulation and wait for the data.
                tick_response = sync_mode.tick(timeout=2.0)
                
                # Data retrieval
                snapshot, image_rgb, image_seg, trailing_image_rgb, trailing_image_seg = tick_response


                # ======================= Car Chasing Section =========================

                # detect car in image with semantic segnmentation camera
                carInTheImage = semantic.IsThereACarInThePicture(trailing_image_seg)

                leading_location = world.player.get_transform()
                trailing_location = trailing_vehicle.get_transform()

                newX, newY = carDetector.CreatePointInFrontOFCar(trailing_location.location.x, trailing_location.location.y,
                                                                     trailing_location.rotation.yaw)

                angle = carDetector.getAngle([trailing_location.location.x, trailing_location.location.y], [newX, newY],
                                                [leading_location.location.x, leading_location.location.y])

                possibleAngle = 0
                drivableIndexes = []

                bbox, predicted_distance,predicted_angle = carDetector.getDistance(world.player, trail_cam_rgb, carInTheImage, extrapolation=extrapolate,nOfFramesToSkip=nOfFramesToSkip)

                if frame % LP_FREQUENCY_DIVISOR == 0:
                    # This is the bottle neck and takes times to run. But it is necessary for chasing around turns
                    predicted_angle, drivableIndexes = semantic.FindPossibleAngle(trailing_image_seg,bbox,predicted_angle) 

                steer, throttle = drivingControlAdvanced.PredictSteerAndThrottle(predicted_distance,predicted_angle,None)

                send_control(trailing_vehicle, throttle, steer, 0)

                real_dist = trailing_location.location.distance(leading_location.location)
                # ==================================================================


                if time_cycle >= 1000.0/cycles:
                    time_cycle = 0.0

                    image_seg.convert(carla.ColorConverter.CityScapesPalette)
                    # ==================================================================
                    # TODO - run features
                    try:
                        traj, lane_mask = get_trajectory_from_lane_detector(ld, image_seg) # stay in lane
                        # dgmd_mask = image_pipeline(image_seg)
                        # save_img(image_seg)
                        print(traj.shape, traj)
                    except:
                        continue
                    # ==================================================================
                    # Debug data
                    # debug_view(image_rgb, image_seg, lane_mask)
                    # debug_view(image_rgb, image_seg)
                    # cv2.imshow("debug view", dgmd_mask)
                    # cv2.waitKey(1)

                # PID Control
                if traj.any():
                    speed = get_speed(world.player)
                    throttle, steer = a_controller.get_control(traj, speed, desired_speed=15, dt=1./FPS)
                    send_control(world.player, throttle, steer, 0)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                frame +=1
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
