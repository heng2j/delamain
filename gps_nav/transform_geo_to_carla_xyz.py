"""
Transforming Geolocation (altitude, longitude, altitude) to Carla Location (x, y, z)

Based on the carla function _location_to_gps
https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py

Adapted by DevGlitch
"""

import math


def from_gps_to_xyz(latitude: float, longitude: float, altitude: float):  # lat_ref, lon_ref
    """Get carla location x y z coordinates from GPS (latitude, longitude, altitude)."""

    # Equatorial mean radius of Earth in meters
    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    earth_radius = 6378137.0

    # Hardcoded can only work for town 01 to 07
    lat_ref = 0.0
    lon_ref = 0.0

    scale = math.cos(lat_ref * math.pi / 180.0)
    base_x = scale * lon_ref * math.pi * earth_radius / 180.0
    base_y = scale * earth_radius * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))

    x = scale * (longitude - base_x) * math.pi * earth_radius / 180.0
    y = (scale * earth_radius * math.log(math.tan((90.0 + latitude) * math.pi / 360.0)) - base_y) * -1
    z = altitude

    # like carla.Location
    location = (x, y, z)
    return location

# For debug
# lat = -0.002647366503438775
# lon = -6.620580971056203e-05
# alt = 0.0
#
# result = from_gps_to_xyz(lat, lon, alt)
# print(result)
#
# lat_dif = result[0] + 7.369997024536133
# lon_dif = result[1] - 294.7034912109375
#
# print("x dif=", lat_dif)
# print("y dif=", lon_dif)
