import numpy as np
from config import *


class KeyPoint:
    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        self.z = None
        self.visibility = None

    @classmethod
    def create_for_face(cls, idx, lm):
        name = 'face{}'.format(idx)
        point = cls(name)
        point._read_landmark(lm)

        return point

    @classmethod
    def create_from_landmark(cls, name, lm):
        name = str(name).split('.')[1]
        point = cls(name)
        point._read_landmark(lm)

        return point

    @staticmethod
    def get_angle(p1, p2, p3):
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _read_landmark(self, lm):
        self.visibility = lm.visibility
        pos = [lm.x, lm.y, lm.z]
        self.x, self.y, self.z = tuple(np.multiply(pos, [WIDTH, HEIGHT, WIDTH]).astype(int))
