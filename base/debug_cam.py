import numpy as np
import cv2

from lane_tracking.lane_track import lane_track_init


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((512, 1024, 4))
    i3 = i2[:, :, :3]
    # image.save_to_disk('data/image_%s.png' % image.timestamp)
    return i3


def lane_track_debug(mask):
    """
    lane color: red
    """
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    rgb_mask = (rgb_mask*255).astype(np.uint8)
    rgb_mask[np.where((rgb_mask <= [1, 1, 1]).all(axis=2))] = [0, 0, 255]
    rgb_mask[np.where((rgb_mask >= [200, 200, 200]).all(axis=2))] = [0, 0, 0]
    return rgb_mask


def debug_view(*sensors):
    """
    param: sensor data

    Steps: Add/Filter/Apply masks
    """
    # TODO: Add masks
    # rgb cam
    rgb_cam = process_img(sensors[0])
    # rgb_mask = 0
    rgb_mask = lane_track_debug(sensors[1])

    debug_image = rgb_cam[:]
    # TODO: Filter masks
    # lane track condition
    if rgb_mask.any():
        lane_cnd = rgb_mask[:, :, 2] > 10
        debug_image[lane_cnd] = rgb_mask[lane_cnd]

    #########################################################################
    cv2.imshow("debug view", debug_image)
    cv2.waitKey(1)
