import numpy as np
import cv2
from pathlib import Path

from lane_tracking.cores.lane_detection.lane_detector import LaneDetector
from lane_tracking.util.carla_util import carla_img_to_array
from lane_tracking.cores.lane_detection.camera_geometry import CameraGeometry


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((512, 1024, 4))
    i3 = i2[:, :, :3]
    # image.save_to_disk('data/image_%s.png' % image.timestamp)
    cv2.imshow("", i3)
    cv2.waitKey(1)


def lane_track_debug(image):
    cg = CameraGeometry()
    K = cg.intrinsic_matrix

    # ld = LaneDetector(model_path=Path("lane_tracking/best_model.pth").absolute())
    # image_arr = carla_img_to_array(image)
    poly_left, poly_right = ld(image_arr)

    return poly_left, poly_right

def debug_view(*image):
    