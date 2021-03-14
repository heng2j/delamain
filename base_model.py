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

import carla
import random
import time
import cv2
import numpy as np
import math
# import pygame
from matplotlib import pyplot as plt
# from lane_tracking.lane_detector import LaneDetector
from gps_nav.gnss import *


def main():
    actor_list = []
    # model_path = "lane_tracking/best_model.pth"
    # lane_detect = LaneDetector(model_path=model_path)

    def process_img(image):
        i = np.array(image.raw_data)
        i2 = i.reshape((512, 1024, 4))
        i3 = i2[:, :, :3]
        # image.save_to_disk('data/image_%s.png' % image.timestamp)
        cv2.imshow("", i3)
        cv2.waitKey(1)
        # left, right = lane_detect(i3)
        # x = np.linspace(0,60)
        # yl = left(x)
        # yr = right(x)
        # print(yl.shape, yr.shape)
        # # plt.plot(x, yl, label="yl")
        # # plt.plot(x, yr, label="yr")


        # visualize(predicted_mask_left=left,
        #           predicted_mask_right=right)
        # return i3/255.0

    def visualize(**images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()


    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Town05, Town07 has highway, parking, standard roads
        world = client.get_world()
        spectator = world.get_spectator()

        # blueprint library
        blueprint_library = world.get_blueprint_library()

        # Agent = Tesla model 3
        bp = blueprint_library.find('vehicle.dodge_charger.police')

        # Camera RGB sensor
        bp_cam_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_cam_rgb.set_attribute('image_size_x', '1024')
        bp_cam_rgb.set_attribute('image_size_y', '512')
        bp_cam_rgb.set_attribute('fov', '110')
        cam_rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        # bp_cam_rgb.set_attribute('sensor_tick', str(1.0))  # Wait time for sensor to update (1.0 = 1s)

        # GNSS Sensor
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_location = carla.Location(0, 0, 0)
        gnss_rotation = carla.Rotation(0, 0, 0)
        gnss_transform = carla.Transform(gnss_location, gnss_rotation)
        gnss_bp.set_attribute("sensor_tick", str(1.0))  # Wait time for sensor to update (1.0 = 1s)

        # Spawn Agent
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Spawn Sensors
        cam_rgb = world.spawn_actor(bp_cam_rgb, cam_rgb_transform, attach_to=vehicle)
        actor_list.append(cam_rgb)
        print('created %s' % cam_rgb.type_id)
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        actor_list.append(gnss)
        print('created %s' % gnss.type_id)

        # Activate Sensors
        # cam_rgb.listen(lambda data: process_img(data))  # Camera RGB Sensor
        gnss.listen(lambda event: gnss_live_location(event))  # GNSS Sensor

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Spectator set to follow Ego vehicle
        ttl = time.time()
        max_ttl = time.time() + 60  # 60 seconds time to live
        while ttl < max_ttl:
            world.wait_for_tick()
            trans_rot = cam_rgb.get_transform().rotation
            x_mod = -8 * math.cos(math.radians(trans_rot.yaw))
            y_mod = -8 * math.sin(math.radians(trans_rot.yaw))
            trans_loc = cam_rgb.get_transform().location + carla.Location(x=x_mod, y=y_mod, z=3)
            transform = carla.Transform(trans_loc, trans_rot)
            spectator.set_transform(transform)
            ttl = time.time()

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    main()
