import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import torch


######################################################################################
def geo_to_location(geo):
    """
    param: carla.geo(latitude, longitude, altitude)
    return: carla.Location(x, y, z)
    """

    model = torch.load("geo2loc.pth")
    test = [[geo.latitude, geo.longitude, geo.altitude]]
    prediction = model.predict(X=test)
    x = prediction[0][0]
    y = prediction[0][1]
    z = prediction[0][2]
    return carla.Location(x=x,y=y,z=z)
######################################################################################
