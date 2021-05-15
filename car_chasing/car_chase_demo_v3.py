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




# Frenet imports

from cubic_spline_planner import *
from quintic_polynomials_planner import QuinticPolynomial


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    # return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)        # 3.6 * meter per seconds = kmh
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)        # meter per seconds


def generate_target_course(x, y, z):
    csp = Spline3D(x, y, z)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, rz, ryaw, rk = [], [], [], [], []
    for i_s in s:
        ix, iy, iz = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        rz.append(iz)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy, iz = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fz = iz
            fp.x.append(fx)
            fp.y.append(fy)
            fp.z.append(fz)
            

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

#         print("fp.x: ", fp.x)
#         print("fp.yaw: ", fp.yaw)
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist



def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def frenet_to_inertial(s, d, csp):
    """
    transform a point from frenet frame to inertial frame
    input: frenet s and d variable and the instance of global cubic spline class
    output: x and y in global frame
    """
    ix, iy, iz = csp.calc_position(s)
    iyaw = csp.calc_yaw(s)
    x = ix + d * math.cos(iyaw + math.pi / 2.0)
    y = iy + d * math.sin(iyaw + math.pi / 2.0)

    return x, y, iz, iyaw


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx

    print("w_size: ", w_size)
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist
    return f_idx + closest_wp_index

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.ds = []
        self.c = []

        self.v = []  # speed

class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=10.0, K_D=0.0, K_I=0.0):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        if float(0.1) > 0:
            self.dt = float(0.1)
        else:
            self.dt = 0.05
        self._e_buffer = deque(maxlen=10)

    def reset(self):
        self._e_buffer = deque(maxlen=10)

    def run_step(self, target_speed):
        """
        Execute one step of longitudinal control to reach a given target speed.
        :param target_speed: target speed in m/s
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        return self._pid_control(target_speed, current_speed), current_speed

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations
        :param target_speed:  target speed in m/s
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self.dt) + (self._K_I * _ie * self.dt), 0.0, 1.0)


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=0.2, K_D=0.0, K_I=0.0):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        if float(0.1) > 0:
            self.dt = float(0.1)
        else:
            self.dt = 0.05
        self._e_buffer = deque(maxlen=10)

        self.prev_prop = np.nan
        self.prev_prev_prop = np.nan
        self.curr_prop = np.nan
        self.deriv_list = []
        self.deriv_len = 5

    def reset(self):
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint [x, y]
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint[0] -
                          v_begin.x, waypoint[1] -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self.dt) + (self._K_I * _ie * self.dt), -1.0, 1.0)

    def run_step_2_wp(self, waypoint1, waypoint2):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control_2_wp(waypoint1, waypoint2, self._vehicle.get_transform())

    def _pid_control_2_wp(self, waypoint1, waypoint2, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint [x, y]
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint2[0] -
                          waypoint1[0], waypoint2[1] -
                          waypoint1[1], 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self.dt) + (self._K_I * _ie * self.dt), -1.0, 1.0)


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        if not args_lateral:
            args_lateral = {'K_P': 0.3, 'K_D': 0.0, 'K_I': 0.0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 40.0, 'K_D': 0.1, 'K_I': 4}

        self._vehicle = vehicle
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    # def reset(self):
    #     self._lon_controller.reset()
    #     self._lat_controller.reset()
    #     control = carla.VehicleControl()
    #     control.steer = 0.0
    #     control.throttle = 0.0
    #     control.brake = 1.0
    #     control.hand_brake = True
    #     control.manual_gear_shift = False
    #     self._vehicle.apply_control(control)

    def run_step_2_wp(self, target_speed, waypoint1, waypoint2):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.
        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle, speed = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step_2_wp(waypoint1, waypoint2)
        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 0.8  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 10 #7  # maximum road width [m]
D_ROAD_W = 1.0 # 1.0  # road width sampling length [m]
DT = 0.2 #0.2 # time tick [s]
MAX_T = 6.0  # max prediction time [m]
MIN_T = 3.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.3  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0


show_animation = True


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

        def look_ahead_local_planer(waypoints: List, current_idx: int=0, look_ahead: int=20 ):
            
            wx = []
            wy = []
            wz = []

            for p in waypoints[current_idx: (current_idx + look_ahead)]:
                wp = carla.Transform(carla.Location(p[0] ,p[1],p[2]),carla.Rotation(p[3],p[4],p[5]))
                wx.append(wp.location.x)
                wy.append(wp.location.y)
                wz.append(wp.location.z)

#             print("wx: ", wx)
            tx, ty, tyaw, tc, csp = generate_target_course(wx, wy, wz)
            
            return csp
        

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

                # Advance the simulation and wait for the data. 
                # tick_response = sync_mode.tick(timeout=2.0)





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
                # vehicleController = VehiclePIDController(vehicle, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})

                path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

                # f_idx = 1





                ob = np.array([ [actor.get_transform().location.x,actor.get_transform().location.y] for actor in actor_list if actor.id in other_actor_ids ])



                # wps_to_go = len(path.t) - 3 
                # print("wps_to_go: ", wps_to_go)


                # # follows path until end of WPs 
                # loop_counter = 0
                # while f_idx < wps_to_go:
                    
                #     pre_f_idx = copy.deepcopy(f_idx)

                #     loop_counter += 1
                #     print("f_idx before : ", f_idx)

                #     vehicleToFollow_location = [vehicle.get_location().x, vehicle.get_location().y, math.radians(vehicle.get_transform().rotation.yaw)]
                #     f_idx = closest_wp_idx(vehicleToFollow_location, path, f_idx)
                #     print("f_idx after: ", f_idx)
                #     if pre_f_idx == f_idx:
                #         f_idx += 1
                #     if f_idx > wps_to_go:
                #         print("f_idx is greater than wps_to_go")
                #         f_idx -=1
         
                #     cmdSpeed = math.sqrt((path.s_d[f_idx]) ** 2 + (path.d_d[f_idx]) ** 2)
                #     cmdWP = [path.x[f_idx], path.y[f_idx]]
                #     cmdWP2 = [path.x[f_idx + 1], path.y[f_idx + 1]]


                #     # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
                #     control = vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
                #     vehicle.apply_control(control)  # apply control
                            
                #     # # Set new way points
                #     # new_vehicleToFollow_transform = carla.Transform()
                #     # new_vehicleToFollow_transform.rotation =  carla.Rotation(pitch=0.0, yaw=math.degrees(path.yaw[f_idx]), roll=0.0) 
        
                #     # new_vehicleToFollow_transform.location.x = path.x[f_idx]
                #     # new_vehicleToFollow_transform.location.y = path.y[f_idx]
                #     # new_vehicleToFollow_transform.location.z = path.z[f_idx]


                #     # wp = (new_vehicleToFollow_transform.location.x,  new_vehicleToFollow_transform.location.y,  new_vehicleToFollow_transform.location.z, 
                #     # new_vehicleToFollow_transform.rotation.pitch, new_vehicleToFollow_transform.rotation.yaw, new_vehicleToFollow_transform.rotation.roll )

                #     # leading_waypoints.append(wp)

                #     # vehicleToFollow.set_transform(new_vehicleToFollow_transform)


                # cmdSpeed = math.sqrt((path.s_d[1]) ** 2 + (path.d_d[1]) ** 2)
                # cmdWP = [path.x[1], path.y[1]]
                # cmdWP2 = [path.x[1 + 1], path.y[1 + 1]]


                # # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
                # control = vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
                # vehicle.apply_control(control)  # apply control




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


                # if frame % LP_FREQUENCY_DIVISOR == 0:
                #     # Update vehicleToFollow transorm with obsticles
                #     # --------------------------------------------------------------
                #     _LOOKAHEAD_INDEX = 5
                #     _BP_LOOKAHEAD_BASE = 8.0              # m 
                #     _BP_LOOKAHEAD_TIME = 2.0              # s 


                #     # unsupported operand type(s) for +: 'float' and 'Vector3D'
                #     lookahead_time = _BP_LOOKAHEAD_BASE +  _BP_LOOKAHEAD_TIME *  vehicleToFollow.get_velocity().z


                #     location3 = obsticle_vehicle.get_transform()
                    
                #     # Calculate the goal state set in the local frame for the local planner.
                #     # Current speed should be open loop for the velocity profile generation.
                #     ego_state = [location2.location.x, location2.location.y, location2.rotation.yaw, vehicleToFollow.get_velocity().z]

                #     # Set lookahead based on current speed.
                #     b_planner.set_lookahead(_BP_LOOKAHEAD_BASE + _BP_LOOKAHEAD_TIME * vehicleToFollow.get_velocity().z)

                    
                #     # Perform a state transition in the behavioural planner.
                #     b_planner.transition_state(evaluation.history, ego_state, current_speed)
                #     # print("The current speed = %f" % current_speed)

                #     # # Find the closest index to the ego vehicle.
                #     # closest_len, closest_index = behavioural_planner.get_closest_index(evaluation.history, ego_state)

                #     # print("closest_len: ", closest_len)
                #     # print("closest_index: ", closest_index)
                    
                #     # # Find the goal index that lies within the lookahead distance
                #     # # along the waypoints.
                #     # goal_index = b_planner.get_goal_index(evaluation.history, ego_state, closest_len, closest_index)

                #     # print("goal_index: ", goal_index)

                #     # # Set goal_state
                #     # goal_state = evaluation.history[goal_index]
            
                #     # Compute the goal state set from the behavioural planner's computed goal state.
                #     goal_state_set = l_planner.get_goal_state_set(b_planner._goal_index, b_planner._goal_state, evaluation.history, ego_state)

                #     # # Calculate planned paths in the local frame.
                #     paths, path_validity = l_planner.plan_paths(goal_state_set)

                #     # # Transform those paths back to the global frame.
                #     paths = local_planner.transform_paths(paths, ego_state)

                #     # Detect obsticle car
                #     obsticle_bbox, obsticle_predicted_distance, obsticle_predicted_angle = carDetector.getDistance(obsticle_vehicle, camera_rgb,carInTheImage,extrapolation=extrapolate,nOfFramesToSkip=nOfFramesToSkip)

                #     obsticle_bbox =[ [bbox[0],bbox[1]] for bbox in obsticle_bbox] 
     
                #     print("paths: ", paths)


                #     if obsticle_bbox:
                #         # # Perform collision checking.
                #         collision_check_array = l_planner._collision_checker.collision_check(paths, [obsticle_bbox])
                #         print("collision_check_array: ", collision_check_array)

                #         # Compute the best local path.
                #         best_index = l_planner._collision_checker.select_best_path_index(paths, collision_check_array, b_planner._goal_state)
                #         print("The best_index :", best_index)


                #     desired_speed = b_planner._goal_state[2]
                #     print("The desired_speed = %f" % desired_speed)

                # newX, newY = carDetector.CreatePointInFrontOFCar(location2.location.x, location2.location.y,location2.rotation.yaw)
                # new_angle = carDetector.getAngle([location2.location.x, location2.location.y], [newX, newY],
                #                              [location3.location.x, location3.location.y])
                
                # tmp = evaluation.history[counter-1]
                # currentPos = carla.Transform(carla.Location(tmp[0] + 0 ,tmp[1],tmp[2]),carla.Rotation(tmp[3],tmp[4],tmp[5]))
                # vehicleToFollow.set_transform(currentPos)


                # --------------------------------------------------------------





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

#                             tx, new_ty, tyaw, tc, trailing_csp = generate_target_course(wx, wy, wz)


                        trail_path =  frenet_optimal_planning(trailing_csp, trail_s0, trail_c_speed, trail_c_d, trail_c_d_d, trail_c_d_dd, ob)

                        if trail_path:

                            trail_s0 = trail_path.s[1]
                            trail_c_d = trail_path.d[1]
                            trail_c_d_d = trail_path.d_d[1]
                            trail_c_d_dd = trail_path.d_dd[1]
                            trail_c_speed = trail_path.s_d[1]

    #                     elif len(trail_path.x) < 2:

    #                         trail_s0 = 0
    #                     else :
    #                         trail_s0 = 0


                            new_vehicle_transform = carla.Transform()
                            new_vehicle_transform.rotation =  carla.Rotation(pitch=0.0, yaw=math.degrees(trail_path.yaw[1]), roll=0.0) 

                            new_vehicle_transform.location.x = trail_path.x[1]
                            new_vehicle_transform.location.y = trail_path.y[1]
                            new_vehicle_transform.location.z = trail_path.z[1]

#                                 location2 = new_vehicle_transform
                            vehicle.set_transform(new_vehicle_transform)







                location1 = vehicle.get_transform()
                location2 = vehicleToFollow.get_transform()


                # # Update vehicle position by detecting vehicle to follow position
                # newX, newY = carDetector.CreatePointInFrontOFCar(location1.location.x, location1.location.y,location1.rotation.yaw)
                # angle = carDetector.getAngle([location1.location.x, location1.location.y], [newX, newY],
                #                              [location2.location.x, location2.location.y])

                # possibleAngle = 0
                # drivableIndexes = []
                # bbox = []

                
                # bbox, predicted_distance,predicted_angle = carDetector.getDistance(vehicleToFollow, camera_rgb, carInTheImage,extrapolation=extrapolate,nOfFramesToSkip=nOfFramesToSkip)

                # # if frame % LP_FREQUENCY_DIVISOR == 0:
                # #     # This is the bottle neck and takes times to run. But it is necessary for chasing around turns
                # #     predicted_angle, drivableIndexes = semantic.FindPossibleAngle(image_segmentation,bbox,predicted_angle) # This is still necessary need to optimize it 
                    
                # steer, throttle = drivingControlAdvanced.PredictSteerAndThrottle(predicted_distance,predicted_angle,None)

                # # # This is a new method
                # # send_control(vehicle, throttle, steer, 0)


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

                # DrawDrivable(drivableIndexes, image_segmentation.width // 10, image_segmentation.height // 10, display)


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

        drivesFileNames = ['ride5.p']  #  ['ride5.p']   ['ride8.p']  ['ride10.p']  for testing advance angle turns # turnel ['ride15.p']  

        for fileName in drivesFileNames:
            main(optimalDistance=optimalDistance,followDrivenPath=followDrivenPath,chaseMode=chaseMode, evaluateChasingCar=evaluateChasingCar,driveName=os.path.join(drivesDir,fileName),record=record,followMode=followMode)


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
