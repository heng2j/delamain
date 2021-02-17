import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


image_fn = str(Path("data/carla_scene.png"))
image = cv2.imread(image_fn)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
# print(image.shape)

boundary_fn = image_fn.replace(".png", "_boundary.txt")
boundary_gt = np.loadtxt(boundary_fn)
# exploit that in the Carla world coordinates the road is mostly in the Xw-Yw plane
print("Zw-coords: ", boundary_gt[:,2])
plt.plot(boundary_gt[:,0], boundary_gt[:,1], label="left lane boundary")
plt.plot(boundary_gt[:,3], boundary_gt[:,4], label="right lane boundary")
plt.axis("equal")
plt.legend();

trafo_fn = image_fn.replace(".png", "_trafo.txt")
trafo_world_to_cam = np.loadtxt(trafo_fn)


import sys
from pathlib import Path
sys.path.append(str(Path('../../')))
import camera_geometry

cg = camera_geometry.CameraGeometry()
K = cg.intrinsic_matrix

for boundary_polyline in [boundary_gt[:,0:3], boundary_gt[:,3:]]:
    uv = camera_geometry.project_polyline(boundary_polyline, trafo_world_to_cam, K)
    u,v = uv[:,0], uv[:,1]
    plt.plot(u,v)
plt.imshow(image)

def create_label_img(lb_left, lb_right):
    label = np.zeros((512, 1024, 3))
    colors = [[1, 1, 1], [2, 2, 2]]
    for color, lb in zip(colors, [lb_left, lb_right]):
        cv2.polylines(label, np.int32([lb]), isClosed=False, color=color, thickness=5)
    label = np.mean(label, axis=2)  # collapse color channels to get gray scale
    return label

uv_left = camera_geometry.project_polyline(boundary_gt[:,0:3], trafo_world_to_cam, K)
uv_right = camera_geometry.project_polyline(boundary_gt[:,3:], trafo_world_to_cam, K)

label = create_label_img(uv_left, uv_right)
plt.imshow(label, cmap="gray");
plt.show()
cv2.imwrite("mylabel.png", label)