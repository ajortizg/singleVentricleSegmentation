import os.path as osp
import numpy as np
import sys
import torch


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import plots
from utilities.deprecated import torch_utils
from utilities.quaternary_transforms import rotx, roty, rotz


class Ellipse:
    def __init__(self, cx, cy, rx, ry, angle):
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        self.angle = angle

        self.compute()

    def compute(self):
        theta = torch.deg2rad(torch.arange(0.0, 360.0, 1.0))
        self.x = self.rx * torch.cos(theta)
        self.y = self.ry * torch.sin(theta)
        rad_ang = torch.radians(self.angle)
        R = torch.array([
            [torch.cos(rad_ang), -torch.sin(rad_ang)],
            [torch.sin(rad_ang), torch.cos(rad_ang)],
        ])

        self.x, self.y = torch.dot(R, torch.array([self.x, self.y]))
        self.x += self.cx
        self.y += self.cy
        self.xy = torch.array([self.x, self.y])

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
    def __init__(self, cx, cy, cz, rx, ry, rz, angx, angy, angz):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.angx = angx
        self. angy = angy
        self.angz = angz
        self.bins = 200
        # self.color = np.random.rand(3)

        # Only for visualization
        # self.compute()

    def compute(self):
        u = np.linspace(0, 2 * np.pi, self.bins)
        v = np.linspace(0, np.pi, self.bins)
        self.x = self.rx * np.outer(np.cos(u), np.sin(v))
        self.y = self.ry * np.outer(np.sin(u), np.sin(v))
        self.z = self.rz * np.outer(np.ones_like(u), np.cos(v))

        self.rotate()

        self.x += self.cx
        self.y += self.cy
        self.z += self.cz

        self.xyz = np.array([self.x, self.y, self.z])

        # self.pc = np.zeros((3, np.size(self.x)))
        # self.pc[0, :] = np.reshape(self.x, -1)
        # self.pc[1, :] = np.reshape(self.y, -1)
        # self.pc[2, :] = np.reshape(self.z, -1)

    def create_voxels(self, grid, constant, value1, value2):
        self.constant = constant
        self.value1 = value1
        self.value2 = value2
        xx = grid[:, :, :, 0]
        yy = grid[:, :, :, 1]
        zz = grid[:, :, :, 2]
        NZ, NY, NX, _ = grid.shape
        CZ, CY, CX = NZ // 2, NY // 2, NX // 2

        self.mask = (xx - self.cx - CX) ** 2 / self.rx**2 + (yy - self.cy - CY) ** 2 \
            / self.ry**2 + (zz - self.cz - CZ) ** 2 / self.rz**2 <= 1.0

        # Rotation about the image center
        Rx = rotx(self.angx)
        Ry = roty(self.angy)
        Rz = rotz(self.angz)
        Rot = Rz @ Ry @ Rx
        Rot = np.linalg.inv(Rot)
        tx = CX - Rot[0, 0] * CX - Rot[0, 1] * CY - Rot[0, 2] * CZ
        ty = CY - Rot[1, 0] * CX - Rot[1, 1] * CY - Rot[1, 2] * CZ
        tz = CZ - Rot[2, 0] * CX - Rot[2, 1] * CY - Rot[2, 2] * CZ
        t = np.array([tx, ty, tz], dtype=np.float)

        grid_t = np.zeros((NZ, NY, NX, 3), dtype=np.float)
        for z in range(NZ):
            for y in range(NY):
                for x in range(NX):
                    grid_t[z, y, x, :] = Rot @ grid[z, y, x, :] + t

        self.voxels = np.zeros((NZ, NY, NX))

        if not constant:
            # Gray value linealy varing in x direction
            gray = np.linspace(value1, value2, NX)
            for x in range(NX):
                self.voxels[:, :, x] = gray[x] * self.mask[:, :, x]
        else:
            self.voxels = np.where(self.mask, value1, 0.0)

        self.voxels = torch_utils.warp(
            torch.from_numpy(self.voxels).unsqueeze(dim=0).unsqueeze(dim=0),
            torch.from_numpy(grid_t).unsqueeze(dim=0),
            mode="bilinear").squeeze().detach().cpu().numpy()
        self.mask = np.where(self.voxels > 0, True, False)

        if not constant:
            # Gray value linealy varing in x direction
            gray = np.linspace(value1, value2, NX)
            for x in range(NX):
                self.voxels[:, :, x] = gray[x] * self.mask[:, :, x]
        else:
            self.voxels = np.where(self.mask, value1, 0.0)

    def rotate(self):
        Rx = rotx(self.angx)
        Ry = roty(self.angy)
        Rz = rotz(self.angz)
        Rot = Rz @ Ry @ Rx
        self.xyz = np.array([self.x, self.y, self.z])

        for i in range(self.bins):
            for j in range(self.bins):
                self.xyz[:, i, j] = Rot @ self.xyz[:, i, j]

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
