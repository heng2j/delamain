import numpy as np
import cv2
import carla

from lane_tracking.lane_track import lane_track_init


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((288, 768, 4))
    i3 = i2[:, :, :3]
    # image.save_to_disk('data/image_%s.png' % image.timestamp)
    return i3


def save_img(image):
    image.save_to_disk('data/dgmd/image_%s.png' % image.timestamp)


def process_seg(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    return process_img(image)


def fast_track_debug(mask):
    """
    lane color: red
    """
    rgb_mask = mask
    return rgb_mask


def debug_view(*sensors):
    """
    param: sensor data

    Steps: Add/Filter/Apply masks
    """
    # TODO: Add masks
    # rgb cam
    rgb_cam = process_img(sensors[0])
    # seg cam
    seg_cam = process_seg(sensors[1])
    # lane_mask
    if sensors[2].any():
        lane_mask = sensors[2]
    debug_image = rgb_cam[:]
    # TODO: Filter masks
    # lane track condition
    if lane_mask.any():
        lane_cnd = lane_mask[:, :] < 0.1
        debug_image[lane_cnd, 0] = 255

    #########################################################################
    cv2.imshow("debug view", debug_image)
    cv2.waitKey(1)
