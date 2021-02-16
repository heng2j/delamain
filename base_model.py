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


def main():
    actor_list = []

    def process_img(image):
        i = np.array(image.raw_data)
        i2 = i.reshape((640, 480, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)
        # return i3/255.0


    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Town05, Town07 has highway, parking, standard roads
        world = client.get_world()
        spectator = world.get_spectator()

        # blueprint library
        blueprint_library = world.get_blueprint_library()

        # Agent = Tesla model 3
        bp = blueprint_library.find('vehicle.tesla.model3')

        # Camera RGB sensor
        bp_cam_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_cam_rgb.set_attribute('image_size_x', '640')
        bp_cam_rgb.set_attribute('image_size_y', '480')
        bp_cam_rgb.set_attribute('fov', '110')
        # bp_cam_rgb.set_attribute('sensor_tick', '1.0')

        # Spawn Agent
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Spawn Sensors
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        cam_rgb = world.spawn_actor(bp_cam_rgb, transform, attach_to=vehicle)
        actor_list.append(cam_rgb)
        print('created %s' % cam_rgb.type_id)

        # Activate Sensors
        # cam_rgb.listen(lambda data: process_img(data))

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
