import glob
import os
import sys

try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from typing import NamedTuple, List
import random
import cv2
from pathlib import Path
import numpy as np
import pygame
import math
import pickle
import copy
from collections import deque

from lane_tracking.lane_track import get_speed, send_control

from car_chasing.util.carla_util import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, find_weather_presets, get_font, should_quit #draw_image
from car_chasing.util.geometry_util import dist_point_linestring

# For birdeye view 
from carla_birdeye_view import (
    BirdViewProducer,
    BirdView,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    BirdViewCropType,
)
from carla_birdeye_view.mask import PixelDimensions


import imageio
from copy import deepcopy
def draw_image(surface, image, image2, location1, location2, blend=False, record=False,driveName='',smazat=[]):
    if record:
        driveName = driveName.split('/')[1]
        dirName = os.path.join('output',driveName)
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        # image.save_to_disk(dirName+'/%07d' % image.frame)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def DrawDrivable(indexes, w, h, display):
    if len(indexes) != 0:
        BB_COLOR = (11, 102, 35)
        for i in range(10):
            for j in range(10):
                if indexes[i*10+j] == 1:
                    pygame.draw.line(display, BB_COLOR, (j*w,i*h) , (j*w+w,i*h))
                    pygame.draw.line(display, BB_COLOR, (j*w,i*h), (j*w,i*h+h))
                    pygame.draw.line(display, BB_COLOR, (j*w+w,i*h), (j*w+w,i*h+h))
                    pygame.draw.line(display, BB_COLOR,  (j*w,i*h+h), (j*w+w,i*h+h))


# New Classes
class Evaluation():
    def __init__(self):
        self.sumMAE = 0
        self.sumRMSE = 0
        self.n_of_frames = 0
        self.n_of_collisions = 0
        self.history = []

    def AddError(self, distance, goalDistance):
        self.n_of_frames += 1
        self.sumMAE += abs(goalDistance-distance)
        self.sumRMSE += abs(goalDistance-distance)*abs(goalDistance-distance)

    def WriteIntoFileFinal(self, filename, driveName):
        if self.n_of_frames > 0:
            self.sumMAE = self.sumMAE / float(self.n_of_frames)
            self.sumRMSE = self.sumRMSE / float(self.n_of_frames)

        with open(filename,'a') as f:
            f.write(str(driveName)+', '+str(self.sumMAE)+', '+str(self.sumRMSE)+', '+str(self.n_of_collisions)+'\n')

    def LoadHistoryFromFile(self, fileName):
        self.history = pickle.load( open(fileName, "rb"))

    def CollisionHandler(self,event):
        self.n_of_collisions += 1



# Frenet imports
from car_chasing.cubic_spline_planner import *
from car_chasing.quintic_polynomials_planner import QuinticPolynomial
from car_chasing.dynamic_frenet import * 


show_animation = True


def main(optimalDistance, followDrivenPath, chaseMode, evaluateChasingCar, driveName='',record=False, followMode=False,
         resultsName='results',P=None,I=None,D=None,nOfFramesToSkip=0):

    # New imports
    from car_chasing.DrivingControl import DrivingControl
    from car_chasing.DrivingControlAdvanced import DrivingControlAdvanced
    from car_chasing.CarDetector import CarDetector
    from car_chasing.SemanticSegmentation import SemanticSegmentation


    # New Variables
    extrapolate = True
    optimalDistance = 8
    followDrivenPath = True
    evaluateChasingCar = True
    record = False
    chaseMode = True
    followMode = False
    counter = 1
    sensors = []

    vehicleToFollowSpawned = False
    obsticle_vehicleSpawned = False

    # New objects
    carDetector = CarDetector()
    drivingControl = DrivingControl(optimalDistance=optimalDistance)
    drivingControlAdvanced = DrivingControlAdvanced(optimalDistance=optimalDistance)
    evaluation = Evaluation()
    semantic = SemanticSegmentation()


    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(80.0)

    world = client.get_world()
    weather_presets = find_weather_presets()
    world.set_weather(weather_presets[3][0])

    # Set BirdView
    birdview_producer = BirdViewProducer(
        client,
        PixelDimensions(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT),
        pixels_per_meter=4,
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        render_lanes_on_junctions=False,
    )


    try:
        m = world.get_map()

        blueprint_library = world.get_blueprint_library()

        veh_bp = random.choice(blueprint_library.filter('vehicle.dodge_charger.police'))
        vehicle = world.spawn_actor(
            veh_bp,
            m.get_spawn_points()[90])
        actor_list.append(vehicle)

        # New vehicle property
        vehicle.set_simulate_physics(True)

        if followDrivenPath:
            evaluation.LoadHistoryFromFile(driveName)
            first = evaluation.history[0]
            start_pose = carla.Transform(carla.Location(first[0], first[1], first[2]),
                                        carla.Rotation(first[3], first[4], first[5]))
            vehicle.set_transform(start_pose)

        # New Sensors
        
        # front cam for object detection
        camera_rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_rgb_blueprint.set_attribute('fov', '90')
        camera_rgb = world.spawn_actor(
           camera_rgb_blueprint,
            carla.Transform(carla.Location(x=1.5, z=1.4,y=0.3), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors.append(camera_rgb)

            
        # segmentation camera
        camera_segmentation = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=1.5, z=1.4,y=0), carla.Rotation(pitch=0)), #5,3,0 # -0.3
            attach_to=vehicle)
        actor_list.append(camera_segmentation)
        sensors.append(camera_segmentation)


        
        # Set up local planner and behavnioural planner
        # --------------------------------------------------------------
        # Planning Constants
        NUM_PATHS = 7
        BP_LOOKAHEAD_BASE      = 8.0              # m
        BP_LOOKAHEAD_TIME      = 2.0              # s
        PATH_OFFSET            = 1.5              # m
        CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
        CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
        TIME_GAP               = 1.0              # s
        PATH_SELECT_WEIGHT     = 10
        A_MAX                  = 1.5              # m/s^2
        SLOW_SPEED             = 2.0              # m/s
        STOP_LINE_BUFFER       = 3.5              # m
        LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
        LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the 
                                                # local planner operate at a lower
                                                # frequency than the controller
                                                # (which operates at the simulation
                                                # frequency). Must be a natural
                                                # number.

        PREV_BEST_PATH         = []
        stopsign_fences = [] 

        # --------------------------------------------------------------
        frame = 0
        max_error = 0
        FPS = 30
        speed = 0
        cross_track_error = 0
        start_time = 0.0
        times = 8
        LP_FREQUENCY_DIVISOR = 4

        # ==============================================================================
        # -- Frenet related stuffs ---------------------------------------
        # ==============================================================================

        # -----------------Set Obstical Positions  -----------------------

        # TMP obstacle lists
        ob = np.array([
                       [233.980630, 50.523910],
                       [232.980630, 80.523910],
                       [234.980630, 100.523910],
                       [235.786942, 110.530586],
                       [234.980630, 120.523910],
                       ])
        

        # -----------------Set way points  -----------------------
        wx = []
        wy = []
        wz = []

        for p in evaluation.history:
            wp = carla.Transform(carla.Location(p[0] ,p[1],p[2]),carla.Rotation(p[3],p[4],p[5]))
            wx.append(wp.location.x)
            wy.append(wp.location.y)
            wz.append(wp.location.z)


        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy, wz)
        
        old_tx = deepcopy(tx)
        old_ty = deepcopy(ty)

        

        # Leading waypoints 
        leading_waypoints = []


        # other actors
        other_actor_ids = []


        # Trailing
        trail_path = None        
        real_dist = 0 
        
        
                        
        # initial state
        c_speed = 10.0 / 3.6  # current speed [m/s]
        c_d = 2.0  # current lateral position [m]
        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current latral acceleration [m/s]
        s0 = 0.0  # current course position    
        
        trail_c_speed = 20.0 / 3.6  # current speed [m/s]
        trail_c_d = 2.0  # current lateral position [m]
        trail_c_d_d = 0.0  # current lateral speed [m/s]
        trail_c_d_dd = 0.0  # current latral acceleration [m/s]
        trail_s0 = 0.0  # current course position    
        
                
        i = 0
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                start_time += clock.get_time()

                # snapshot, image_rgb, image_segmentation = tick_response
                snapshot, img_rgb, image_segmentation = sync_mode.tick(timeout=2.0)

                # detect car in image with semantic segnmentation camera
                carInTheImage = semantic.IsThereACarInThePicture(image_segmentation)

                line = []

                current_speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))

                # Spawn a vehicle to follow
                if not vehicleToFollowSpawned and followDrivenPath:
                    vehicleToFollowSpawned = True
                    location1 = vehicle.get_transform()
                    newX, newY = carDetector.CreatePointInFrontOFCar(location1.location.x, location1.location.y,
                                                                     location1.rotation.yaw)
                    diffX = newX - location1.location.x
                    diffY = newY - location1.location.y
                    newX = location1.location.x - (diffX*5)
                    newY = location1.location.y - (diffY*5)

                    start_pose.location.x = newX
                    start_pose.location.y = newY

                    vehicle.set_transform(start_pose)

                    start_pose2 = random.choice(m.get_spawn_points())

                    bp = blueprint_library.filter('model3')[0]
                    bp.set_attribute('color', '0,101,189')
                    vehicleToFollow = world.spawn_actor(
                        bp,
                        start_pose2)

                    start_pose2 = carla.Transform()
                    start_pose2.rotation = start_pose.rotation

                    start_pose2.location.x = start_pose.location.x
                    start_pose2.location.y = start_pose.location.y
                    start_pose2.location.z = start_pose.location.z

                    vehicleToFollow.set_transform(start_pose2)

                    actor_list.append(vehicleToFollow)
                    vehicleToFollow.set_simulate_physics(True)
                    vehicleToFollow.set_autopilot(True)

                if followDrivenPath:
                    if counter >= len(evaluation.history):
                        break
                    tmp = evaluation.history[counter]
                    currentPos = carla.Transform(carla.Location(tmp[0] + 5 ,tmp[1],tmp[2]),carla.Rotation(tmp[3],tmp[4],tmp[5]))
                    vehicleToFollow.set_transform(currentPos)
                    counter += 1


                # ------------------- Set up obsticle vehicle for testing  --------------------------------

                # Set up obsticle vehicle for testing 
  
                if not obsticle_vehicleSpawned and followDrivenPath:
                    obsticle_vehicleSpawned = True
            
                    for obsticle_p in ob:

                        start_pose3 = random.choice(m.get_spawn_points())

                        obsticle_vehicle = world.spawn_actor(
                        random.choice(blueprint_library.filter('jeep')),
                        start_pose3)

                        start_pose3 = carla.Transform()
                        start_pose3.rotation = start_pose2.rotation
                        start_pose3.location.x = obsticle_p[0] 
                        start_pose3.location.y =  obsticle_p[1]
                        start_pose3.location.z =  start_pose2.location.z

                        obsticle_vehicle.set_transform(start_pose3)
                        obsticle_vehicle.set_autopilot(True)

        
                        actor_list.append(obsticle_vehicle)
                        obsticle_vehicle.set_simulate_physics(True)
                
                
                        other_actor_ids.append(obsticle_vehicle.id)
                    
                        
                #---- Leading Frenet -----------------------------
                path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)
                ob = np.array([ [actor.get_transform().location.x,actor.get_transform().location.y] for actor in actor_list if actor.id in other_actor_ids ])

                # --------------------------- Working leading frenet 

                # Set new way points
                new_vehicleToFollow_transform = carla.Transform()
                new_vehicleToFollow_transform.rotation =  carla.Rotation(pitch=0.0, yaw=math.degrees(path.yaw[1]), roll=0.0) 
    
                new_vehicleToFollow_transform.location.x = path.x[1]
                new_vehicleToFollow_transform.location.y = path.y[1]
                new_vehicleToFollow_transform.location.z = path.z[1]


                wp = (new_vehicleToFollow_transform.location.x,  new_vehicleToFollow_transform.location.y,  new_vehicleToFollow_transform.location.z, 
                new_vehicleToFollow_transform.rotation.pitch, new_vehicleToFollow_transform.rotation.yaw, new_vehicleToFollow_transform.rotation.roll )


                vehicleToFollow.set_transform(new_vehicleToFollow_transform)


                s0 = path.s[1]
                c_d = path.d[1]
                c_d_d = path.d_d[1]
                c_d_dd = path.d_dd[1]
                c_speed = path.s_d[1]


                if i > 2:
                    leading_waypoints.append(wp)

                location2 = None

                # ---- Car chasing activate -----
                # Give time for leading car to cumulate waypoints 

                #------- Trailing Frenet --------------------------------------
                # Start frenet once every while
                if (i > 50):
#                     trailing_csp = look_ahead_local_planer(waypoints=leading_waypoints, current_idx=0, look_ahead=len(leading_waypoints))


#                     speed = get_speed(vehicle)
                    FRENET_FREQUENCY_DIVISOR = 1
                    if (frame % FRENET_FREQUENCY_DIVISOR == 0) and (real_dist > 5):


                        wx = []
                        wy = []
                        wz = []

                        for p in leading_waypoints:
                            wp = carla.Transform(carla.Location(p[0] ,p[1],p[2]),carla.Rotation(p[3],p[4],p[5]))
                            wx.append(wp.location.x)
                            wy.append(wp.location.y)
                            wz.append(wp.location.z)


                        tx, ty, tyaw, tc, trailing_csp = generate_target_course(wx, wy, wz)



                        trail_path =  frenet_optimal_planning(trailing_csp, trail_s0, trail_c_speed, trail_c_d, trail_c_d_d, trail_c_d_dd, ob)

                        if trail_path:

                            trail_s0 = trail_path.s[1]
                            trail_c_d = trail_path.d[1]
                            trail_c_d_d = trail_path.d_d[1]
                            trail_c_d_dd = trail_path.d_dd[1]
                            trail_c_speed = trail_path.s_d[1]


                            new_vehicle_transform = carla.Transform()
                            new_vehicle_transform.rotation =  carla.Rotation(pitch=0.0, yaw=math.degrees(trail_path.yaw[1]), roll=0.0) 

                            new_vehicle_transform.location.x = trail_path.x[1]
                            new_vehicle_transform.location.y = trail_path.y[1]
                            new_vehicle_transform.location.z = trail_path.z[1]

                            vehicle.set_transform(new_vehicle_transform)


                location1 = vehicle.get_transform()
                location2 = vehicleToFollow.get_transform()


                speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))

                real_dist = location1.location.distance(location2.location)
             

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                # draw_image(display, image_rgb)
                draw_image(display, img_rgb, image_segmentation,location1, location2,record=record,driveName=driveName,smazat=line)
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
                    font.render('     distance to target: {:.2f} m'.format(real_dist), True, (255, 255, 255)),
                    (8, 66))

                # Display BirdView
                # Input for your model - call it every simulation step
                # returned result is np.ndarray with ones and zeros of shape (8, height, width)
                
                birdview = birdview_producer.produce(agent_vehicle=vehicleToFollow)
                bgr = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)
                # NOTE imshow requires BGR color model
                cv2.imshow("BirdView RGB", bgr)
                cv2.waitKey(1)


                pygame.display.flip()

                frame += 1
                i += 1

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        optimalDistance = 8
        followDrivenPath = True
        evaluateChasingCar = True
        record = False
        chaseMode = True
        followMode = False

        drivesDir = './car_chasing/drives'
        drivesFileNames = os.listdir(drivesDir)
        drivesFileNames.sort()

        drivesFileNames = ['ride5.p']  #  ['ride5.p']   ['ride8.p']  ['ride10.p']  for testing advance angle turns # turnel ['ride15.p']  

        for fileName in drivesFileNames:
            main(optimalDistance=optimalDistance,followDrivenPath=followDrivenPath,chaseMode=chaseMode, evaluateChasingCar=evaluateChasingCar,driveName=os.path.join(drivesDir,fileName),record=record,followMode=followMode)


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
