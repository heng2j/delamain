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

def car_chasing_init(world: carla.World, position: carla.Transform=None, y_offset: int= 5 ) -> carla.Actor:
    """[summary]

    Args:
        world (World): [description]
        y_offset (int): [description]
        position (carla.Transform, optional): [description]. Defaults to None.

    Returns:
        carla.Actor: [description]
    """

    trailing_vehicle = agent_init(world)
    
    # Set position for trailing vehicle to be relative to leading vehicle if it is not given
    if not position:

        lead_vehicle_transfrom = world.player.get_transform()

        start_pose = carla.Transform()
        start_pose.rotation = lead_vehicle_transfrom.rotation
        start_pose.location.x = lead_vehicle_transfrom.location.x 
        start_pose.location.y =  lead_vehicle_transfrom.location.y + y_offset 
        start_pose.location.z =  lead_vehicle_transfrom.location.z

        trailing_vehicle.set_transform(start_pose)

    return trailing_vehicle


def agent_init(world: carla.World) -> carla.Actor:
    """[summary]

    Args:
        world (World): [description]

    Returns:
        carla.Actor: [description]
    """

    blueprint_library = world.world.get_blueprint_library()

    # Set Trailing vehicle
    m = world.world.get_map()

    trailing_vehicle_bp = blueprint_library.filter('vehicle.dodge_charger.police')[0]
    trailing_vehicle = world.world.spawn_actor(
        trailing_vehicle_bp,
        m.get_spawn_points()[90])
    
    trailing_vehicle.set_simulate_physics(True)

    return trailing_vehicle



