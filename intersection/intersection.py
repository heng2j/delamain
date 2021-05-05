from carla.agents.navigation.controller import VehiclePIDController


def intersection_init(world):
    setup_PID(world.player)


def intersection(carla_loc):
    """
    param: carla_loc - junction
    """
    pass


def setup_PID(vehicle):
    """
    intersection PID: uses gps location to calculate control
    param: vehicle
    """
    args_lateral_dict = {
        'K_P': 1.95,
        'K_D': 0.2,
        'K_I': 0.07

        , 'dt': 1.0 / 10.0
    }
    args_long_dict = {
        'K_P': 1,
        'K_D': 0.0,
        'K_I': 0.75
        , 'dt': 1.0 / 10.0
    }
    PID = VehiclePIDController(vehicle, args_lateral=args_lateral_dict, args_longitudinal=args_long_dict)
    return PID
