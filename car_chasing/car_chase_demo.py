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
import pickle
from util.carla_util import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, find_weather_presets, get_font, should_quit #draw_image
from util.geometry_util import dist_point_linestring



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


def get_trajectory_from_lane_detector(ld, image):
    """
    param: ld = lane detector
    image: windshield rgb cam
    return: traj
    """
    image_arr = carla_img_to_array(image)
    poly_left, poly_right = ld(image_arr)
    x = np.arange(-2,60,1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    x += 0.5
    traj = np.stack((x,y)).T
    return traj

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((512, 1024, 4))
    i3 = i2[:, :, :3]
    # image.save_to_disk('data/image_%s.png' % image.timestamp)
    # cv2.imshow("", i3)
    # cv2.waitKey(1)
    return i3

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

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
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

        # cg = CameraGeometry()
        # ld = LaneDetector(model_path=Path("best_model.pth").absolute())
        # #windshield cam
        # cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        # bp = blueprint_library.find('sensor.camera.rgb')
        # bp.set_attribute('image_size_x', str(cg.image_width))
        # bp.set_attribute('image_size_y', str(cg.image_height))
        # bp.set_attribute('fov', str(cg.field_of_view_deg))
        # camera_windshield = world.spawn_actor(
        #     bp,
        #     cam_windshield_transform,
        #     attach_to=vehicle)
        # actor_list.append(camera_windshield)
        # sensors.append(camera_windshield)


        frame = 0
        max_error = 0
        FPS = 30
        speed = 0
        cross_track_error = 0
        start_time = 0.0
        times = 8
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                start_time += clock.get_time()

                # Advance the simulation and wait for the data. 
                # tick_response = sync_mode.tick(timeout=2.0)

                # snapshot, image_rgb, image_segmentation = tick_response
                snapshot, img_rgb, image_segmentation = sync_mode.tick(timeout=2.0)

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
                    vehicleToFollow.set_autopilot(False)

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

                # if not obsticle_vehicleSpawned and followDrivenPath:
                #     obsticle_vehicleSpawned = True
                #     # Adding new obsticle vehicle 

                #     start_pose3 = random.choice(m.get_spawn_points())

                #     obsticle_vehicle = world.spawn_actor(
                #         random.choice(blueprint_library.filter('jeep')),
                #         start_pose3)

                #     start_pose3 = carla.Transform()
                #     start_pose3.rotation = start_pose2.rotation
                #     start_pose3.location.x = start_pose2.location.x 
                #     start_pose3.location.y =  start_pose2.location.y + 50 
                #     start_pose3.location.z =  start_pose2.location.z

                #     obsticle_vehicle.set_transform(start_pose3)


                #     actor_list.append(obsticle_vehicle)
                #     obsticle_vehicle.set_simulate_physics(True)


                # Update vehicle position by detecting vehicle to follow position
                newX, newY = carDetector.CreatePointInFrontOFCar(location1.location.x, location1.location.y,location1.rotation.yaw)
                angle = carDetector.getAngle([location1.location.x, location1.location.y], [newX, newY],
                                             [location2.location.x, location2.location.y])

                possibleAngle = 0
                drivableIndexes = []
                bbox = []

                carInTheImage = semantic.IsThereACarInThePicture(image_segmentation)
                bbox, predicted_distance,predicted_angle = carDetector.getDistance(vehicleToFollow, camera_rgb,carInTheImage,extrapolation=extrapolate,nOfFramesToSkip=nOfFramesToSkip)

                # This is the bottle neck
                # predicted_angle, drivableIndexes = semantic.FindPossibleAngle(image_segmentation,bbox,predicted_angle) # This is still necessary need to optimize it 
                
                steer, throttle = drivingControlAdvanced.PredictSteerAndThrottle(predicted_distance,predicted_angle,None)

                # This is a new method
                send_control(vehicle, throttle, steer, 0)


                velocity1 = vehicle.get_velocity()
                velocity2 = vehicleToFollow.get_velocity()

                speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))

                real_dist = location1.location.distance(location2.location)



                # if start_time >= 1000.0/times:
                #     start_time = 0.0
                #     traj = get_trajectory_from_lane_detector(ld, image_windshield)

                #     # rgb_cam = process_img(image_windshield)
                #     # cv2.imshow("windshield", rgb_cam)
                #     # cv2.waitKey(1)

                #     # # get velocity and angular velocity
                #     # vel = carla_vec_to_np_array(vehicle.get_velocity())
                #     # forward = carla_vec_to_np_array(vehicle.get_transform().get_forward_vector())
                #     # right = carla_vec_to_np_array(vehicle.get_transform().get_right_vector())
                #     # up = carla_vec_to_np_array(vehicle.get_transform().get_up_vector())
                #     # vx = vel.dot(forward)
                #     # vy = vel.dot(right)
                #     # vz = vel.dot(up)
                #     # ang_vel = carla_vec_to_np_array(vehicle.get_angular_velocity())
                #     # w = ang_vel.dot(up)
                #     # print("vx vy vz w {:.2f} {:.2f} {:.2f} {:.5f}".format(vx,vy,vz,w))

                #     # PID control
                #     speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))
                #     throttle, steer = controller.get_control(traj, speed, desired_speed=44, dt=1./FPS)
                #     send_control(vehicle, throttle, steer, 0)


                #     dist = dist_point_linestring(np.array([0,0]), traj)

                #     cross_track_error = int(dist*100)

                # max_error = max(max_error, cross_track_error)

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
                # display.blit(
                #     font.render('     cross track error: {:03d} cm'.format(cross_track_error), True, (255, 255, 255)),
                #     (8, 64))
                # display.blit(
                #     font.render('     max cross track error: {:03d} cm'.format(max_error), True, (255, 255, 255)),
                #     (8, 82))


                # Draw bbox on following vehicle
                if len(bbox) != 0:
                    points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
                    BB_COLOR = (248, 64, 24)
                    # draw lines
                    # base
                    pygame.draw.line(display, BB_COLOR, points[0], points[1])
                    pygame.draw.line(display, BB_COLOR, points[1], points[2])
                    pygame.draw.line(display, BB_COLOR, points[2], points[3])
                    pygame.draw.line(display, BB_COLOR, points[3], points[0])
                    # top
                    pygame.draw.line(display, BB_COLOR, points[4], points[5])
                    pygame.draw.line(display, BB_COLOR, points[5], points[6])
                    pygame.draw.line(display, BB_COLOR, points[6], points[7])
                    pygame.draw.line(display, BB_COLOR, points[7], points[4])
                    # base-top
                    pygame.draw.line(display, BB_COLOR, points[0], points[4])
                    pygame.draw.line(display, BB_COLOR, points[1], points[5])
                    pygame.draw.line(display, BB_COLOR, points[2], points[6])
                    pygame.draw.line(display, BB_COLOR, points[3], points[7])

                # DrawDrivable(drivableIndexes, image_segmentation.width // 10, image_segmentation.height // 10, display)



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
