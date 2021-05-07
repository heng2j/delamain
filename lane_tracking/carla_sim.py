# Code based on Carla examples, which are authored by 
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).

# How to run: 
# cd into the parent directory of the 'code' directory and run
# python -m code.tests.control.carla_sim
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
import random
from pathlib import Path
import numpy as np
import pygame
from util.carla_util import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, find_weather_presets, draw_image, get_font, should_quit
from util.geometry_util import dist_point_linestring
import argparse


def get_trajectory_from_lane_detector(ld, image):
    # get lane boundaries using the lane detector
    image_arr = carla_img_to_array(image)
    poly_left, poly_right = ld(image_arr)
    # trajectory to follow is the mean of left and right lane boundary
    # note that we multiply with -0.5 instead of 0.5 in the formula for y below
    # according to our lane detector x is forward and y is left, but
    # according to Carla x is forward and y is right.
    x = np.arange(-2,60,1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    # hence correct x coordinates
    x += 0.5
    traj = np.stack((x,y)).T
    return traj

def get_trajectory_from_map(m, vehicle):
    # get 80 waypoints each 1m apart. If multiple successors choose the one with lower waypoint.id
    wp = m.get_waypoint(vehicle.get_transform().location)
    wps = [wp]
    for _ in range(20):
        next_wps = wp.next(1.0)
        if len(next_wps) > 0:
            wp = sorted(next_wps, key=lambda x: x.id)[0]
        wps.append(wp)

    # transform waypoints to vehicle ref frame
    traj = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in wps]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

    traj = trafo_matrix_world_to_vehicle @ traj
    traj = traj.T
    traj = traj[:,:2]
    return traj

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)



def main(use_lane_detector=False):
    # Imports
    from cores.lane_detection.lane_detector import LaneDetector
    from cores.lane_detection.camera_geometry import CameraGeometry
    from cores.control.pure_pursuit import PurePursuitPlusPID


    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(80.0)

    #client.load_world('Town06')
    client.load_world('Town05')
    world = client.get_world()
    weather_presets = find_weather_presets()
    # print(weather_presets)
    # world.set_weather(weather_presets[3][0])
    world.set_weather(carla.WeatherParameters.HardRainSunset)

    controller = PurePursuitPlusPID()

    try:
        m = world.get_map()

        blueprint_library = world.get_blueprint_library()

        veh_bp = random.choice(blueprint_library.filter('vehicle.dodge_charger.police'))
        # veh_bp.set_attribute('color','64,81,181')
        vehicle = world.spawn_actor(
            veh_bp,
            m.get_spawn_points()[90])
        actor_list.append(vehicle)


        # visualization cam (no functionality)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]

        
        if use_lane_detector:
            cg = CameraGeometry()

            ld = LaneDetector(model_path=Path(".best_model_multi_dice_loss.pth").absolute())

            #windshield cam
            cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
            bp = blueprint_library.find('sensor.camera.rgb')
            fov = cg.field_of_view_deg
            bp.set_attribute('image_size_x', str(cg.image_width))
            bp.set_attribute('image_size_y', str(cg.image_height))
            bp.set_attribute('fov', str(fov))
            camera_windshield = world.spawn_actor(
                bp,
                cam_windshield_transform,
                attach_to=vehicle)
            actor_list.append(camera_windshield)
            sensors.append(camera_windshield)


        frame = 0
        max_error = 0
        FPS = 30
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                
                # Advance the simulation and wait for the data. 
                tick_response = sync_mode.tick(timeout=2.0)

                if use_lane_detector:
                    snapshot, image_rgb, image_windshield = tick_response
                    traj = get_trajectory_from_lane_detector(ld, image_windshield)
                else:
                    snapshot, image_rgb = tick_response
                    traj = get_trajectory_from_map(m, vehicle)

                # get velocity and angular velocity
                vel = carla_vec_to_np_array(vehicle.get_velocity())
                forward = carla_vec_to_np_array(vehicle.get_transform().get_forward_vector())
                right = carla_vec_to_np_array(vehicle.get_transform().get_right_vector())
                up = carla_vec_to_np_array(vehicle.get_transform().get_up_vector())
                vx = vel.dot(forward)
                vy = vel.dot(right)
                vz = vel.dot(up)
                ang_vel = carla_vec_to_np_array(vehicle.get_angular_velocity())
                w = ang_vel.dot(up)
                print("vx vy vz w {:.2f} {:.2f} {:.2f} {:.5f}".format(vx,vy,vz,w))

                speed = np.linalg.norm( carla_vec_to_np_array(vehicle.get_velocity()))
                throttle, steer = controller.get_control(traj, speed, desired_speed=10, dt=1./FPS)
                send_control(vehicle, throttle, steer, 0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                dist = dist_point_linestring(np.array([0,0]), traj)

                cross_track_error = int(dist*100)
                max_error = max(max_error, cross_track_error)

                # Draw the display.
                draw_image(display, image_rgb)
                display.blit(
                    font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('     FPS (simulated): % 5d ' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('     speed: {:.2f} m/s'.format(speed), True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('     cross track error: {:03d} cm'.format(cross_track_error), True, (255, 255, 255)),
                    (8, 64))
                display.blit(
                    font.render('     max cross track error: {:03d} cm'.format(max_error), True, (255, 255, 255)),
                    (8, 82))

                pygame.display.flip()

                frame += 1
                # if save_gif and frame > 1000:
                #     print("frame=",frame)
                #     imgdata = pygame.surfarray.array3d(pygame.display.get_surface())
                #     imgdata = imgdata.swapaxes(0,1)
                #     if frame < 1200:
                #         images.append(imgdata)
                

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Carla simulation with your control algorithm.')
    parser.add_argument("--ld", action="store_true", help="Use reference trajectory from your LaneDetector class")
    args = parser.parse_args()

    try:
        main(use_lane_detector = args.ld)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
