import carla
import random
import time


def parallel_parking(world, actor_list, vehicle):
    """
    Example of parallel parking maneuvre.
    Could be developed further when an ultrasonic sensor becomes available on CARLA.
    """

    # blueprint library
    blueprint_library = world.get_blueprint_library()

    # Pre-defined parking locations
    parked_locations = [
        carla.Transform(carla.Location(x=22, y=190.20, z=0.05), carla.Rotation(yaw=180)),  # Front vehicle
        carla.Transform(carla.Location(x=35.1, y=190.20, z=0.05), carla.Rotation(yaw=180))  # Back vehicle
        ]

    # Spawning two park vehicles based on parked location
    bp_parked = random.choice(blueprint_library.filter('vehicle.tesla.cybertruck'))
    for pos in parked_locations:
        vehicle_parked = world.spawn_actor(bp_parked, pos)
        actor_list.append(vehicle_parked)

    # Starting manoeuvre
    while vehicle.get_location().x > 23.5:
        vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, reverse=False))
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, reverse=False))
    time.sleep(1)

    while True:
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.3, steer=0.7, brake=0.0, reverse=True))
        time.sleep(0.1)
        if vehicle.get_location().y < 193.3:
            break
    while vehicle.get_location().y < 193.3:
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.2, steer=-0.7, brake=0.0, reverse=True))
        time.sleep(0.1)
        if abs(vehicle.get_transform().rotation.yaw) > 180 - 2:
            vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
            break

    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0, reverse=False))
    time.sleep(1.2)
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, reverse=False))
    time.sleep(0.5)


def perpendicular_parking(world, actor_list, vehicle):
    """
    Example of perpendicular parking maneuvre.
    Could be developed further when an ultrasonic sensor becomes available on CARLA.
    """

    # blueprint library
    blueprint_library = world.get_blueprint_library()

    # Pre-defined parking locations
    parked_locations = [
        carla.Transform(carla.Location(x=218.48, y=59.2, z=0.05), carla.Rotation(yaw=0)),  # Left vehicle
        carla.Transform(carla.Location(x=218.48, y=65.8, z=0.05), carla.Rotation(yaw=0))  # Right vehicle
        ]

    # Spawning two park vehicles based on parked location
    bp_parked = random.choice(blueprint_library.filter('vehicle.tesla.cybertruck'))
    for pos in parked_locations:
        vehicle_parked = world.spawn_actor(bp_parked, pos)
        actor_list.append(vehicle_parked)

    # Starting perpendicular maneuvre
    while vehicle.get_location().y < 57:
        vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, reverse=True))
    while True:
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.3, steer=-0.7, brake=0.0, reverse=True))
        time.sleep(0.1)
        if vehicle.get_location().y > 62.5:
            break
    while vehicle.get_location().x > 220.5:
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.1, steer=0.7, brake=0.0, reverse=True))
        time.sleep(0.2)
    vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0, reverse=True))
    time.sleep(1)
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, reverse=False))
    time.sleep(0.5)
