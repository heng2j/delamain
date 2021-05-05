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


def save_img(image, path='data/dgmd/image_%s.png'):
    image.save_to_disk(path % image.timestamp)


def process_seg(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    return process_img(image)


def fast_track_debug(mask):
    """
    lane color: red
    """
    rgb_mask = mask
    return rgb_mask


def debug_view(*sensors, text=[1.0]):
    """
    param: sensor data

    Steps: Add/Filter/Apply masks
    """
    # TODO: Add masks
    # rgb cam
    rgb_cam = process_img(sensors[0])
    # seg cam
    seg_cam = process_seg(sensors[1])

    debug_image = rgb_cam[:]

    # TODO: Filter masks
    # lane track condition
    if sensors[2].any():
        lane_cnd = sensors[2][:, :] < 0.1
        debug_image[lane_cnd, 2] = 255

    # add text
    debug_image = cv2.putText(np.array(debug_image), 'Road Status: '+str("Junction" if text[0] else "Lane"), (15, 15),
                              cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    debug_image = cv2.putText(np.array(debug_image), 'Control Type: ' + str(text[1]), (15, 30),
                              cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    # debug_image = cv2.putText(np.array(debug_image), 'Route: ' + str(text[2]), (15, 45),
    #                           cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    #########################################################################
    cv2.imshow("debug view", debug_image)
    cv2.waitKey(1)
