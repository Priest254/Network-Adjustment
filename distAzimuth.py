
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:30:52 2024

@author: Allan
"""

import math as m
import numpy as np
import pandas as pd

def getDist(row):
    """
    Compute distance between two points.
    Takes a dictionary row containing the Xs and Ys.
    Returns distance.
    """
    x1, y1, x2, y2 = np.float64([row['X_FROM'], row['Y_FROM'], row['X_TO'], row['Y_TO']])
    dist = m.sqrt(m.pow(x2 - x1, 2) + m.pow(y2 - y1, 2))
    return dist

def getDist2(x1, y1, x2, y2):
    """
    Compute distance between two points.
    Takes Xs and Ys.
    Returns distance.
    """
    dist = m.sqrt(m.pow(x2 - x1, 2) + m.pow(y2 - y1, 2))
    return dist

def getAzimuth(x1, y1, x2, y2):
    """
    Compute azimuth between two points.
    Takes Xs and Ys.
    Returns Azimuth.
    """
    cX, cY = np.float64([(x2 - x1), (y2 - y1)])
    azimuth = 0
    if cX < 0 and cY < 0:
        azimuth = 180 + (m.degrees(m.atan(cX / cY)))
    elif cX < 0 and cY >= 0:
        azimuth = 360 + (m.degrees(m.atan(cX / cY)))
    elif cX >= 0 and cY < 0:
        azimuth = 180 + (m.degrees(m.atan(cX / cY)))
    elif cX >= 0 and cY >= 0:
        azimuth = m.degrees(m.atan(cX / cY))

    return m.radians(azimuth)
