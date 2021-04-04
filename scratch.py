# import glob
# import os
# import sys
#
# try:
#     sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
#
# import carla
#
# client = carla.Client("localhost", 2000)
# client.set_timeout(10.0)
# world = client.get_world()

# weather = carla.WeatherParameters(
#     cloudyness=80.0,
#     precipitation=30.0,
#     sun_altitude_angle=-50.0)
#
# world.set_weather(weather)


# def debug_view(*image):
#     for i in image:
#         print(i)
#     print(image)
#
#
# debug_view(1,2,3,4,5)

import cv2
import numpy as np

from lane_tracking.lane_track import lane_track_init


src = cv2.imread("lane_tracking/data/carla_scene.png")

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((512, 1024, 4))
    i3 = i2[:, :, :3]
    # image.save_to_disk('data/image_%s.png' % image.timestamp)
    # cv2.imshow("", i3)
    # cv2.waitKey(1)
    return i3


def lane_track_debug(image):
    """
    lane color: red
    """
    cg, ld = lane_track_init()
    # mask, left, right = ld.detect(image)
    # rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # rgb_mask = (rgb_mask*255).astype(np.uint8)
    # rgb_mask[np.where((rgb_mask <= [1, 1, 1]).all(axis=2))] = [0, 0, 255]
    # rgb_mask[np.where((rgb_mask >= [200, 200, 200]).all(axis=2))] = [0, 0, 0]
    # return rgb_mask

    left, right = ld.detect_and_fit(image)
    return left, right


def debug_view(*sensors):
    """
    param: sensor data

    Steps: Add/Filter/Apply masks
    """
    # for data in sensors:
    #     rgb_cam = process_img(data)

    # TODO: Add masks
    # rgb cam
    # rgb_cam = process_img(sensors[0])
    rgb_cam = sensors[0]
    # rgb_mask = 0
    # rgb_mask = lane_track_debug(rgb_cam)
    #
    # debug_image = rgb_cam[:]
    # # TODO: Filter masks
    # # lane track condition
    # if rgb_mask.any():
    #     lane_cnd = rgb_mask[:, :, 2] > 10
    #     debug_image[lane_cnd] = rgb_mask[lane_cnd]
    #
    # #########################################################################
    # cv2.imshow("debug view", debug_image)
    # cv2.waitKey(1)

    left, right = lane_track_debug(rgb_cam)
    print(left)
    # cv2.imshow("debug view", left)
    cv2.waitKey()


debug_view(src)
