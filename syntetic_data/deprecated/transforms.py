import numpy as np
import math
import random

# small rotation
# small translation
# compression and gray value change


def rotx(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [1, 0, 0],
        [0, math.cos(rad), -math.sin(rad)],
        [0, math.sin(rad), math.cos(rad)],
    ])


def roty(deg):
    rad = np.deg2rad(deg)
    return np.array([[math.cos(rad), 0, 0], [0, 1, 0],
                     [-math.sin(rad), 0, math.cos(rad)]])


def rotz(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [math.cos(rad), -math.sin(rad), 0],
        [math.sin(rad), math.cos(rad), 0],
        [0, 0, 1],
    ])


def SE3(rot, t):
    T = np.eye(4, 4)
    T[0:3, 0:3] = rot
    T[0:3, 3] = t
    return T


def scale(sx, sy, sz):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])


def rot2d(deg):
    rad = np.deg2rad(deg)
    return np.array([[math.cos(rad), -math.sin(rad)],
                     [math.sin(rad), math.cos(rad)]])


# T2 = SE3(roty(67)*rotx(7)*rotz(43), [5, 6, 7])
# print(T2)
