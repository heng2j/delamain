import carla
import numpy as np
import matplotlib as plt
import math


def gnss_live_location(event):
    """Get the GNSS Measurements
    :return: Values of Altitude, Latitude, and Longitude
    :rtype: float
    """
    alt = event.altitude
    lat = event.latitude
    lon = event.longitude
    # print("GNSS measure:\n" + str(event) + '\n')  # For full details of measurements incl. frame and timestamp
    print("altitude:", str(alt))
    print("latitude:", str(lat))
    print("longitude:", str(lon), "\n")
    return alt, lat, lon


# Actual location
# vehicle_location = ...

# Destination location
# ...

# Total t to destination
# dist_to_dest = ...
# print(dist_to_dest)

# while True:
#     # Update gnss location
#     ...
#
#     # Update distance to destination
#     updated_dist_to_dest = ...
#
#     #
#     ...
