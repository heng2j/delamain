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
import cv2
from pathlib import Path
import numpy as np
import pygame
import math
import weakref
import pickle
from util.carla_util import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, find_weather_presets, get_font, should_quit #draw_image
from util.geometry_util import dist_point_linestring


# For birdeye view
from carla_birdeye_view import (
    BirdViewProducer,
    BirdView,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    BirdViewCropType,
)
from carla_birdeye_view.mask import PixelDimensions

# For planners
from object_avoidance import local_planner
from object_avoidance import behavioural_planner

from frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from cubic_spline_planner import *

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



def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    # return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)        # 3.6 * meter per seconds = kmh
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)        # meter per seconds



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


class LineOfSightSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.distance = None
        self.vehicle_ahead = None
        self._parent = parent_actor
        # self.sensor_transform = carla.Transform(carla.Location(x=4, z=1.7), carla.Rotation(yaw=0)) # Put this sensor on the windshield of the car.
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance', '200')
        bp.set_attribute('hit_radius', '0.5')
        bp.set_attribute('only_dynamics', 'True')
        bp.set_attribute('debug_linetrace', 'True')
        bp.set_attribute('sensor_tick', '0.0')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LineOfSightSensor._on_los(weak_self, event))

    def reset(self):
        self.vehicle_ahead = None
        self.distance = None

    def destroy(self):
        self.sensor.destroy()

    def get_vehicle_ahead(self):
        return self.vehicle_ahead

    # Only works for CARLA 9.6 and above!
    def get_los_distance(self):
        return self.distance

    @staticmethod
    def _on_los(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.vehicle_ahead = event.other_actor
        self.distance = event.distance




def main(optimalDistance, followDrivenPath, chaseMode, evaluateChasingCar, driveName='',record=False, followMode=False,
         resultsName='results',P=None,I=None,D=None,nOfFramesToSkip=0):
    # Imports
    # from cores.lane_detection.lane_detector import LaneDetector
    # from cores.lane_detection.camera_geometry import CameraGeometry
    # from cores.control.pure_pursuit import PurePursuitPlusPID

    # New imports
    from DrivingControl import DrivingControl
    from DrivingControlAdvanced import DrivingControlAdvanced
    from CarDetector import CarDetector
    from SemanticSegmentation import SemanticSegmentation


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

    # Comment out for not displaying pygame window
    # display = pygame.display.set_mode(
    #     (800, 600),
    #     pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(80.0)

    #client.load_world('Town06')
    # client.load_world('Town04')
    world = client.get_world()
    weather_presets = find_weather_presets()
    # print(weather_presets)
    world.set_weather(weather_presets[3][0])
    # world.set_weather(carla.WeatherParameters.HardRainSunset)

    # controller = PurePursuitPlusPID()

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
        collision_sensor = world.spawn_actor(blueprint_library.find('sensor.other.collision'),
                                                carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(lambda event: evaluation.CollisionHandler(event))
        actor_list.append(collision_sensor)

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
        # generate and save global route if it does not exist in the road_maps folder
        global_route = np.empty((0, 3))

        print("Saving global_route...")

        distance = 1
        for p in evaluation.history: #range(1520):

            wp = carla.Transform(carla.Location(p[0] ,p[1],p[2]),carla.Rotation(p[3],p[4],p[5]))

            # wp2 = world.get_map().get_waypoint(carla.Location(x=406, y=-100, z=0.1),
            #                                                 project_to_road=True).next(distance=distance)[0]

            distance += 2
            global_route = np.append(global_route,
                                            [[wp.location.x, wp.location.y,
                                            wp.location.z]], axis=0)
        #     # To visualize point clouds
        #     self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
        np.save('./road_maps/global_route_town05', global_route)

        print("Done saving global_route")


        init_s = 50
        init_d = 0 * 3.5 # self.LANE_WIDTH # reset can be np.random.randint(-1, 3) * self.LANE_WIDTH

        motionPlanner = MotionPlanner()


        # Start Modules
        motionPlanner.start(global_route)
        motionPlanner.reset(init_s, init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)

        # self.world_module.update_global_route_csp(motionPlanner.csp)
        # self.traffic_module.update_global_route_csp(motionPlanner.csp)
        # self.module_manager.start_modules()

        f_idx = 0
        f_state = [init_s, init_d, 0, 0, 0, 0]

        # --------------------------------------------------------------


        frame = 0
        max_error = 0
        FPS = 30
        speed = 0
        cross_track_error = 0
        start_time = 0.0
        times = 8
        LP_FREQUENCY_DIVISOR   = 8                # Frequency divisor to make the 
                                                # local planner operate at a lower
                                                # frequency than the controller
                                                # (which operates at the simulation
                                                # frequency). Must be a natural
                                                # number.


        actors_batch = []



        # Add frenet parameters


        # wx = []
        # wy = []
        
        # for p in evaluation.history:
        #     wp = carla.Transform(carla.Location(p[0] ,p[1],p[2]),carla.Rotation(p[3],p[4],p[5]))
        #     wx.append(wp.location.x)
        #     wy.append(wp.location.y)
        
                
        # tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)


        # # initial state
        # c_speed = 10.0 / 3.6  # current speed [m/s]
        # c_d = 2.0  # current lateral position [m]
        # c_d_d = 0.0  # current lateral speed [m/s]
        # c_d_dd = 0.0  # current latral acceleration [m/s]
        # s0 = 0.0  # current course position    
        

                
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                start_time += clock.get_time()

                # Advance the simulation and wait for the data. 
                # tick_response = sync_mode.tick(timeout=2.0)

                # Display BirdView
                # Input for your model - call it every simulation step
                # returned result is np.ndarray with ones and zeros of shape (8, height, width)
                
                birdview = birdview_producer.produce(agent_vehicle=vehicle)
                bgr = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)
                # NOTE imshow requires BGR color model
                cv2.imshow("BirdView RGB", bgr)
                cv2.waitKey(1)


                # snapshot, image_rgb, image_segmentation = tick_response
                snapshot, img_rgb, image_segmentation = sync_mode.tick(timeout=2.0)

                # detect car in image with semantic segnmentation camera
                carInTheImage = semantic.IsThereACarInThePicture(image_segmentation)

                line = []

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
                    # vehicleToFollow.set_autopilot(False)

                if followDrivenPath:
                    if counter >= len(evaluation.history):
                        break
                    tmp = evaluation.history[counter]
                    currentPos = carla.Transform(carla.Location(tmp[0] ,tmp[1],tmp[2]),carla.Rotation(tmp[3],tmp[4],tmp[5]))
                    vehicleToFollow.set_transform(currentPos)
                    counter += 1


                # Set up obsticle vehicle for testing 
                location1 = vehicle.get_transform()
                location2 = vehicleToFollow.get_transform()

                if not obsticle_vehicleSpawned and followDrivenPath:
                    obsticle_vehicleSpawned = True
                    # Adding new obsticle vehicle 

                    start_pose3 = random.choice(m.get_spawn_points())

                    obsticle_vehicle = world.spawn_actor(
                        random.choice(blueprint_library.filter('jeep')),
                        start_pose3)

                    start_pose3 = carla.Transform()
                    start_pose3.rotation = start_pose2.rotation
                    start_pose3.location.x = start_pose2.location.x 
                    start_pose3.location.y =  start_pose2.location.y + 50 
                    start_pose3.location.z =  start_pose2.location.z

                    obsticle_vehicle.set_transform(start_pose3)


                    actor_list.append(obsticle_vehicle)
                    obsticle_vehicle.set_simulate_physics(True)

                    # Adding obsticle_vehicle to actors_batch
                    los_sensor = LineOfSightSensor(obsticle_vehicle)

                    target_speed = get_speed(obsticle_vehicle)

                    # Get S and D for obsticle_vehicle




                    # cruiseControl = CruiseControl(obsticle_vehicle, los_sensor, self.module_manager, targetSpeed=targetSpeed)

                    # actors_batch.append({'Actor': obsticle_vehicle, 'Sensor': los_sensor, 'Cruise Control': cruiseControl, 'Frenet State': [s, d]})


                # if frame % LP_FREQUENCY_DIVISOR == 0:


                tmp = evaluation.history[counter-1]
                currentPos = carla.Transform(carla.Location(tmp[0] + 0 ,tmp[1],tmp[2]),carla.Rotation(tmp[3],tmp[4],tmp[5]))
                vehicleToFollow.set_transform(currentPos)



                """
                        **********************************************************************************************************************
                        *********************************************** Motion Planner *******************************************************
                        **********************************************************************************************************************
                """

                max_s = 3000 #int(cfg.CARLA.MAX_S)
                targetSpeed = 13.89

                temp = [vehicleToFollow.get_velocity(),vehicleToFollow.get_acceleration()]
                speed = get_speed(vehicleToFollow)
                acc_vec = vehicleToFollow.get_acceleration()
                acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
                psi = math.radians(vehicleToFollow.get_transform().rotation.yaw)
                ego_state = [vehicleToFollow.get_location().x, vehicleToFollow.get_location().y, speed, acc, psi, temp,max_s]


                # motionPlanner.run_step(ego_state, f_idx, None, target_speed=targetSpeed)
                
                # # fpath = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action[0], Tf=5, Vf_n=action[1])
                # fpath, fplist, best_path_idx = motionPlanner.run_step(ego_state, f_idx, self.traffic_module.actors_batch, target_speed=targetSpeed)
                # # wps_to_go = len(fpath.t) - 3    # -2 bc len gives # of items not the idx of last item + 2wp controller is used
                # # f_idx = 1






                # --------------------------------------------------------------


                # Update vehicle position by detecting vehicle to follow position
                newX, newY = carDetector.CreatePointInFrontOFCar(location1.location.x, location1.location.y,location1.rotation.yaw)
                angle = carDetector.getAngle([location1.location.x, location1.location.y], [newX, newY],
                                             [location2.location.x, location2.location.y])

                possibleAngle = 0
                drivableIndexes = []
                bbox = []

                
                bbox, predicted_distance,predicted_angle = carDetector.getDistance(vehicleToFollow, camera_rgb,carInTheImage,extrapolation=extrapolate,nOfFramesToSkip=nOfFramesToSkip)

                if frame % LP_FREQUENCY_DIVISOR == 0:
                    # This is the bottle neck and takes times to run. But it is necessary for chasing around turns
                    predicted_angle, drivableIndexes = semantic.FindPossibleAngle(image_segmentation,bbox,predicted_angle) # This is still necessary need to optimize it 
                    
                steer, throttle = drivingControlAdvanced.PredictSteerAndThrottle(predicted_distance,predicted_angle,None)

                # This is a new method
                send_control(vehicle, throttle, steer, 0)


                speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))

                real_dist = location1.location.distance(location2.location)
             

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                # draw_image(display, image_rgb)
                # draw_image(display, img_rgb, image_segmentation,location1, location2,record=record,driveName=driveName,smazat=line)
                # display.blit(
                #     font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)),
                #     (8, 10))
                # display.blit(
                #     font.render('     FPS (simulated): % 5d ' % fps, True, (255, 255, 255)),
                #     (8, 28))
                # display.blit(
                #     font.render('     speed: {:.2f} m/s'.format(speed), True, (255, 255, 255)),
                #     (8, 46))
                # display.blit(
                #     font.render('     cross track error: {:03d} cm'.format(cross_track_error), True, (255, 255, 255)),
                #     (8, 64))
                # display.blit(
                #     font.render('     max cross track error: {:03d} cm'.format(max_error), True, (255, 255, 255)),
                #     (8, 82))


                # # Draw bbox on following vehicle
                # if len(bbox) != 0:
                #     points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
                #     BB_COLOR = (248, 64, 24)
                #     # draw lines
                #     # base
                #     pygame.draw.line(display, BB_COLOR, points[0], points[1])
                #     pygame.draw.line(display, BB_COLOR, points[1], points[2])
                #     pygame.draw.line(display, BB_COLOR, points[2], points[3])
                #     pygame.draw.line(display, BB_COLOR, points[3], points[0])
                #     # top
                #     pygame.draw.line(display, BB_COLOR, points[4], points[5])
                #     pygame.draw.line(display, BB_COLOR, points[5], points[6])
                #     pygame.draw.line(display, BB_COLOR, points[6], points[7])
                #     pygame.draw.line(display, BB_COLOR, points[7], points[4])
                #     # base-top
                #     pygame.draw.line(display, BB_COLOR, points[0], points[4])
                #     pygame.draw.line(display, BB_COLOR, points[1], points[5])
                #     pygame.draw.line(display, BB_COLOR, points[2], points[6])
                #     pygame.draw.line(display, BB_COLOR, points[3], points[7])

                # # DrawDrivable(drivableIndexes, image_segmentation.width // 10, image_segmentation.height // 10, display)

                print("vehicleToFollow.get_transform()", vehicleToFollow.get_transform())

                # pygame.display.flip()


                frame += 1

        cv2.destroyAllWindows()


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

        drivesDir = './drives'
        drivesFileNames = os.listdir(drivesDir)
        drivesFileNames.sort()

        drivesFileNames = ['ride5.p']  #   ['ride8.p']  ['ride10.p']  for testing advance angle turns # turnel ['ride15.p']  

        for fileName in drivesFileNames:
            main(optimalDistance=optimalDistance,followDrivenPath=followDrivenPath,chaseMode=chaseMode, evaluateChasingCar=evaluateChasingCar,driveName=os.path.join(drivesDir,fileName),record=record,followMode=followMode)


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
