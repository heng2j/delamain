# Code based on Carla examples, which are authored by
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).


import sys
import glob
import os

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import os
import carla
import random
import pygame
import numpy as np
import cv2
from datetime import datetime


from lane_tracking.util.carla_util import (
    carla_vec_to_np_array,
    CarlaSyncMode,
    find_weather_presets,
    draw_image,
    get_font,
    should_quit,
)
from lane_tracking.cores.lane_detection.camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    CameraGeometry,
)
from lane_tracking.util.seg_data_util import mkdir_if_not_exist


store_files = True
town_string = "Town03"
cg = CameraGeometry()
width = cg.image_width
height = cg.image_height

now = datetime.now()
date_time_string = now.strftime("%m_%d_%Y_%H_%M_%S")


def plot_map(m):
    import matplotlib.pyplot as plt

    wp_list = m.generate_waypoints(2.0)
    loc_list = np.array(
        [carla_vec_to_np_array(wp.transform.location) for wp in wp_list]
    )
    plt.scatter(loc_list[:, 0], loc_list[:, 1])
    plt.show()


def random_transform_disturbance(transform):
    lateral_noise = np.random.normal(0, 0.3)
    lateral_noise = np.clip(lateral_noise, -0.3, 0.3)

    lateral_direction = transform.get_right_vector()
    x = transform.location.x + lateral_noise * lateral_direction.x
    y = transform.location.y + lateral_noise * lateral_direction.y
    z = transform.location.z + lateral_noise * lateral_direction.z

    yaw_noise = np.random.normal(0, 5)
    yaw_noise = np.clip(yaw_noise, -10, 10)

    pitch = transform.rotation.pitch
    yaw = transform.rotation.yaw + yaw_noise
    roll = transform.rotation.roll

    return carla.Transform(
        carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll)
    )


# def create_wp(world_map, vehicle):
#     # waypoint = world_map.get_waypoint(
#     #     vehicle.get_transform().location,
#     #     project_to_road=True,
#     #     lane_type=carla.LaneType.Driving,
#     # )
#     # if not waypoint.is_junction:
#     #     waypoint = waypoint.next(1.0)
#     waypoint_list = world_map.generate_waypoints(2.0)
#     print(len(waypoint_list))


def is_junction_label(wp):
    forward_wp = wp.next(20.0)[0]
    back_wp = wp.previous(10.0)[0]
    if forward_wp.is_junction and back_wp.is_junction and wp.is_junction:
        return 1.0
    else:
        return 0.0


def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def save_img(image, path, raw=False):
    array = carla_img_to_array(image)
    if raw:
        np.save(path, array)
    else:
        cv2.imwrite(path, array)


def save_label(label, path):
    np.savetxt(path, label)


def get_random_spawn_point(m):
    pose = random.choice(m.get_spawn_points())
    return m.get_waypoint(pose.location)


data_folder = os.path.join("intersection", "data")


def main():
    mkdir_if_not_exist(data_folder)
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    client.load_world(town_string)
    world = client.get_world()

    try:
        m = world.get_map()
        # plot_map(m)
        start_pose = random.choice(m.get_spawn_points())
        spawn_waypoint = m.get_waypoint(start_pose.location)

        # set weather to sunny
        weather_preset, weather_preset_str = find_weather_presets()[0]
        weather_preset_str = weather_preset_str.replace(" ", "_")
        world.set_weather(weather_preset)
        simulation_identifier = (
            town_string + "_" + weather_preset_str + "_" + date_time_string
        )

        # create a vehicle
        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter("vehicle.dodge_charger.police")),
            start_pose,
        )
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # create camera and attach to vehicle
        cam_rgb_transform = carla.Transform(
            carla.Location(x=0.7, z=cg.height),
            carla.Rotation(pitch=-1 * cg.pitch_deg),
        )
        bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        fov = cg.field_of_view_deg
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        camera_rgb = world.spawn_actor(
            bp, cam_rgb_transform, attach_to=vehicle
        )
        actor_list.append(camera_rgb)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            frame = 0
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_seg = sync_mode.tick(timeout=2.0)
                image_seg.convert(carla.ColorConverter.CityScapesPalette)

                # set label
                waypoint_label = is_junction_label(spawn_waypoint)

                # Choose the next spawn_waypoint and update the car location.
                # ----- change lane with low probability
                if np.random.rand() > 0.9:
                    shifted = None
                    if spawn_waypoint.lane_change == carla.LaneChange.Left:
                        shifted = spawn_waypoint.get_left_lane()
                    elif spawn_waypoint.lane_change == carla.LaneChange.Right:
                        shifted = spawn_waypoint.get_right_lane()
                    elif spawn_waypoint.lane_change == carla.LaneChange.Both:
                        if np.random.rand() > 0.5:
                            shifted = spawn_waypoint.get_right_lane()
                        else:
                            shifted = spawn_waypoint.get_left_lane()
                    if shifted is not None:
                        spawn_waypoint = shifted

                # ----- randomly change yaw and lateral position
                spawn_transform = random_transform_disturbance(
                    spawn_waypoint.transform
                )
                vehicle.set_transform(spawn_transform)

                # Draw the display.
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                draw_image(display, image_seg)
                display.blit(
                    font.render(
                        "% 5d FPS (real)" % clock.get_fps(),
                        True,
                        (255, 255, 255),
                    ),
                    (8, 10),
                )
                display.blit(
                    font.render(
                        "% 5d FPS (simulated)" % fps, True, (255, 255, 255)
                    ),
                    (8, 28),
                )
                display.blit(
                    font.render(
                        "% 5d Waypoint Label" % waypoint_label, True, (255, 255, 255)
                    ),
                    (8, 46),
                )

                spawn_waypoint = get_random_spawn_point(m)

                if store_files:
                    filename_base = simulation_identifier + "_frame_{}".format(
                        frame
                    )
                    if np.random.rand() > 0.1:
                        x_bin_name = "x"
                        label_bin_name = "x_label"
                    else:
                        x_bin_name = "val"
                        label_bin_name = "val_label"

                    # image
                    image_out_path = os.path.join(
                        data_folder, x_bin_name, filename_base + ".png"
                    )
                    save_img(image_seg, image_out_path)
                    # label
                    label_path = os.path.join(
                        data_folder, label_bin_name, filename_base + "_label.txt"
                    )
                    save_label(
                        np.array([waypoint_label]),
                        label_path,
                    )

                pygame.display.flip()
                frame += 1

    finally:
        print("destroying actors.")
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print("done.")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
