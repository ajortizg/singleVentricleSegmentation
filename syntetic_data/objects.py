from abc import abstractmethod
import numpy as np


class Object:
    def __init__(self, cx, cy, cz):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.voxels = None

    @abstractmethod
    def create_voxels(self, grid):
        self.xx = grid[:, :, :, 0]
        self.yy = grid[:, :, :, 1]
        self.zz = grid[:, :, :, 2]


class Sphere(Object):
    def __init__(self, cx, cy, cz, r):
        super().__init__(cx, cy, cz)
        self.r = r

    def create_voxels(self, grid):
        super().create_voxels(grid)
        self.voxels = (self.xx - self.cx)**2 + (self.yy - self.cy)**2 + \
            (self.zz - self.cz)**2 < self.r**2


class Cube(Object):
    def __init__(self, cx, cy, cz, lx, ly, lz):
        super().__init__(cx, cy, cz)
        self.lx = lx
        self.ly = ly
        self.lz = lz

    def create_voxels(self, grid):
        super().create_voxels(grid)
        # self.voxels = (self.xx < self.cx - self.lx//2 | self.xx > self.cx + self.lx//2) & \
        #     (self.yy < self.cy - self.ly//2 | self.yy > self.cy + self.ly//2) & \
        #     (self.zz < self.cz - self.lz//2 | self.zz > self.cz + self.lz//2)

        self.voxels = np.full(self.xx.shape, False)
        self.voxels[self.cz - self.lz//2: self.cz + self.lz//2,
                    self.cy - self.ly//2: self.cy + self.ly//2,
                    self.cx - self.lx//2: self.cx + self.lx//2] = True


class Ellipsoid(Object):
    def __init__(self, cx, cy, cz, rx, ry, rz):
        super().__init__(cx, cy, cz)
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def create_voxels(self, grid):
        super().create_voxels(grid)
        NZ, NY, NX = grid.shape[:3]

        self.voxels = (self.xx - self.cx)**2/self.rx**2 + (self.yy - self.cy)**2/self.ry**2 + \
            (self.zz - self.cz)**2/self.rz**2 <= 1.0
