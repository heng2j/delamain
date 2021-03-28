
from pathlib import Path

from lane_tracking.cores.lane_detection.camera_geometry import CameraGeometry
from lane_tracking.cores.lane_detection.lane_detector import LaneDetector


def lane_track_init():
    cg = CameraGeometry()
    ld = LaneDetector(model_path=Path("lane_tracking/best_model.pth").absolute())
    return cg, ld
