import numpy as np
from transforms import rotx, roty, rotz


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


class Ellipsoid:
    def __init__(self, cx, cy, cz, rx, ry, rz, angx, angy, angz, bins=100):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.angx = angx
        self. angy = angy
        self.angz = angz
        self.bins = bins

        self.compute()

    def compute(self):
        u = np.linspace(0, 2*np.pi, self.bins)
        v = np.linspace(0, np.pi, self.bins)

        self.x = self.rx * np.outer(np.cos(u), np.sin(v))
        self.y = self.ry * np.outer(np.sin(u), np.sin(v))
        self.z = self.rz * np.outer(np.ones_like(u), np.cos(v))

        self.rotate()

        self.x += self.cx
        self.y += self.cy
        self.z += self.cz

        self.xyz = np.array([self.x, self.y, self.z])

    def rotate(self):
        Rx = rotx(self.angx)
        Ry = roty(self.angy)
        Rz = rotz(self.angz)
        Rot = Rz@Ry@Rx
        self.xyz = np.array([self.x, self.y, self.z])

        for i in range(self.bins):
            for j in range(self.bins):
                self.xyz[:, i, j] = Rot@self.xyz[:, i, j]

        self.x = self.xyz[0, :, :]
        self.y = self.xyz[1, :, :]
        self.z = self.xyz[2, :, :]

    def __sub__(self, other):
        return Ellipsoid(self.cx - other.cx,
                         self.cy - other.cy,
                         self.cz - other.cz,
                         self.rx - other.rx,
                         self.ry - other.ry,
                         self.rz - other.rz,
                         self.angx - other.angx,
                         self.angy - other.angy,
                         self.angz - other.angz)

    def __add__(self, other):
        return Ellipsoid(self.cx + other.cx,
                         self.cy + other.cy,
                         self.cz + other.cz,
                         self.rx + other.rx,
                         self.ry + other.ry,
                         self.rz + other.rz,
                         self.angx + other.angx,
                         self.angy + other.angy,
                         self.angz + other.angz)

    def __mul__(self, scalar):
        return Ellipsoid(self.cx * scalar,
                         self.cy * scalar,
                         self.cz * scalar,
                         self.rx * scalar,
                         self.ry * scalar,
                         self.rz * scalar,
                         self.angx * scalar,
                         self.angy * scalar,
                         self.angz * scalar)

    def __str__(self):
        return f"cx: {self.cx:.2f}, cy: {self.cy:.2f}, cz: {self.cz:.2f}, rx: {self.rx:.2f}, ry: {self.ry:.2f}, rz: {self.rz:.2f}, angx: {self.angx:.2f}, angy: {self.angy:.2f}, angz: {self.angz:.2f}"
