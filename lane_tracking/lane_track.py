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
from pathlib import Path
import numpy as np

from lane_tracking.cores.lane_detection.camera_geometry import CameraGeometry
from lane_tracking.cores.lane_detection.lane_detector import LaneDetector
from lane_tracking.util.carla_util import carla_vec_to_np_array, carla_img_to_array


def lane_track_init():
    cg = CameraGeometry()
    ld = LaneDetector(model_path=Path("lane_tracking/best_model_multi_dice_loss.pth").absolute())
    return cg, ld


def get_trajectory_from_lane_detector(ld, image):
    """
    polyfit excepts means junction
    param: ld = lane detector
    image: windshield rgb cam
    return: traj
    """
    image_arr = carla_img_to_array(image)
    try:
        poly_left, poly_right, debug = ld(image_arr)
        x = np.arange(-2,60,1.0)
        y = -0.5*(poly_left(x)+poly_right(x))
        x += 0.5
        traj = np.stack((x,y)).T
        warning = False
    except:
        warning = True
        traj = np.array([])
        debug = np.array([])
    return traj, debug, warning


def get_speed(vehicle):
    return np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))


def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)
