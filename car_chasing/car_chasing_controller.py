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

# Car chasing imports
# New imports
from car_chasing.DrivingControl import DrivingControl
from car_chasing.DrivingControlAdvanced import DrivingControlAdvanced
from car_chasing.CarDetector import CarDetector
from car_chasing.SemanticSegmentation import SemanticSegmentation

class ChaseControl():
    """[summary]
    """

    
    def __init__(self, optimalDistance=8, nOfFramesToSkip=0, extrapolate=True, behaviour_planner_frequency_divisor=2 ):
        """[summary]

        Args:
            extrapolate ([type]): [description]
            behaviour_planner_frequency_divisor ([type]): [description]
            optimalDistance ([type], optional): [description]. Defaults to None.
            nOfFramesToSkip ([type], optional): [description]. Defaults to None.
        """
        
        
        self.extrapolate = extrapolate
        self.nOfFramesToSkip = nOfFramesToSkip
        self.behaviour_planner_frequency_divisor = behaviour_planner_frequency_divisor

        self.carDetector = CarDetector()
        self.drivingControl = DrivingControl(optimalDistance=optimalDistance)
        self.drivingControlAdvanced = DrivingControlAdvanced(optimalDistance=optimalDistance)
        self.semantic = SemanticSegmentation()



    def behaviour_planner(self, leading_vehicle: carla.Vehicle, trailing_vehicle: carla.Vehicle, trailing_image_seg: carla.Image, trail_cam_rgb: carla.Sensor, frame: int):
        """[summary]

        Args:
            leading_vehicle (carla.Vehicle): [description]
            trailing_vehicle (carla.Vehicle): [description]
            trailing_image_seg (carla.Image): [description]
            trail_cam_rgb (carla.Sensor): [description]

        Returns:
            [type]: [description]
        """
        # detect car in image with semantic segnmentation camera
        #  Car detection module 
        carInTheImage = self.semantic.IsThereACarInThePicture(trailing_image_seg)

        leading_location = leading_vehicle.get_transform()
        trailing_location = trailing_vehicle.get_transform()

        newX, newY = self.carDetector.CreatePointInFrontOFCar(trailing_location.location.x, trailing_location.location.y,
                                                                trailing_location.rotation.yaw)

        angle = self.carDetector.getAngle([trailing_location.location.x, trailing_location.location.y], [newX, newY],
                                        [leading_location.location.x, leading_location.location.y])

        possibleAngle = 0
        drivableIndexes = []

        bbox, predicted_distance, predicted_angle = self.carDetector.getDistance(leading_vehicle, trail_cam_rgb, carInTheImage, extrapolation=self.extrapolate,nOfFramesToSkip=self.nOfFramesToSkip)

        if frame % self.behaviour_planner_frequency_divisor == 0:
            # This is the bottle neck and takes times to run. But it is necessary for chasing around turns
            predicted_angle, drivableIndexes = self.semantic.FindPossibleAngle(trailing_image_seg, bbox, predicted_angle) 

        # Car detection module
        steer, throttle = self.drivingControlAdvanced.PredictSteerAndThrottle(predicted_distance, predicted_angle, None)
        real_dist = trailing_location.location.distance(leading_location.location)
        return steer, throttle, real_dist
