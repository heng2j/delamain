import carla
import numpy as np
import matplotlib as plt
import math


def gnss_live_location(event):
    """Get and store the GNSS Measurements.
    :return: Float values of Altitude, Latitude, and Longitude
    """
    alt = event.altitude  # meters
    lat = event.latitude  # degrees
    lon = event.longitude  # degrees

    # print("GNSS measure:\n" + str(event) + '\n')  # For full details of measurements incl. frame and timestamp
    print("altitude:", str(alt))
    print("latitude:", str(lat))
    print("longitude:", str(lon), "\n")

    return alt, lat, lon


# ORIENTATION OF THE CAR???

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
