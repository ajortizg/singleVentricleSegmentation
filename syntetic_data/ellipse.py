import numpy as np


class Ellipse:
    def __init__(self, cx, cy, rx, ry, angle):
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        self.angle = angle

        self.compute()

    def compute(self):
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        self.x = self.rx * np.cos(theta)
        self.y = self.ry * np.sin(theta)
        rad_ang = np.radians(self.angle)
        R = np.array([
            [np.cos(rad_ang), -np.sin(rad_ang)],
            [np.sin(rad_ang), np.cos(rad_ang)],
        ])

        self.x, self.y = np.dot(R, np.array([self.x, self.y]))
        self.x += self.cx
        self.y += self.cy
        self.xy = np.array([self.x, self.y])

    def __sub__(self, other):
        return Ellipse(self.cx - other.cx,
                       self.cy - other.cy,
                       self.rx - other.rx,
                       self.ry - other.ry,
                       self.angle - other.angle)

    def __add__(self, other):
        return Ellipse(self.cx + other.cx,
                       self.cy + other.cy,
                       self.rx + other.rx,
                       self.ry + other.ry,
                       self.angle + other.angle)

    def __mul__(self, scalar):
        return Ellipse(self.cx * scalar,
                       self.cy * scalar,
                       self.rx * scalar,
                       self.ry * scalar,
                       self.angle * scalar)

    def __str__(self):
        return f"cx: {self.cx:.2f}, cy: {self.cy:.2f}, rx: {self.rx:.2f}, ry: {self.ry:.2f}, angle: {self.angle:.2f}"
