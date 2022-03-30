from abc import abstractmethod
import numpy as np
import torch.nn.functional as F
import torch


def scale_grid(grid):
    # scale grid to [-1,1]
    if grid.dim() == 4:
        NZ, NY, NX, _ = grid.shape
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(NX-1, 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(NY-1, 1) - 1.0
        grid[:, :, :, 2] = 2.0 * grid[:, :, :, 2] / max(NZ-1, 1) - 1.0
    elif grid.dim() == 3:
        NY, NX, _ = grid.shape
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(NX-1, 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(NY-1, 1) - 1.0

    return grid


def normalize(x):
    # Normalize between 0 and 1
    min = np.amin(x)
    max = np.amax(x)
    return (x-min)/(max-min)


class Object:
    def __init__(self, cx, cy, cz):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.voxels = None
        self.gray_values = None

    @abstractmethod
    def create_voxels(self, grid):
        if grid.ndim == 4:
            self.xx = grid[:, :, :, 0]
            self.yy = grid[:, :, :, 1]
            self.zz = grid[:, :, :, 2]
        elif grid.ndim == 3:
            self.xx = grid[:, :, 0]
            self.yy = grid[:, :, 1]

    def warp(self, grid, of):
        if self.gray_values is not None:
            grid = torch.from_numpy(grid)
            of = torch.from_numpy(of)

            new_coords = scale_grid(grid - of)
            new_coords.unsqueeze_(dim=0)
            vol = torch.from_numpy(self.gray_values)
            vol.unsqueeze_(dim=0).unsqueeze_(dim=0)

            vol_w = F.grid_sample(vol, new_coords, align_corners=True,
                                  mode="bilinear", padding_mode="zeros")
            return vol_w.squeeze().numpy()
        else:
            return None


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

    def create_voxels(self, grid, constant=False, value=1.0):
        super().create_voxels(grid)
        NZ, NY, NX = grid.shape[:3]

        self.voxels = (self.xx - self.cx)**2/self.rx**2 + (self.yy - self.cy)**2/self.ry**2 + \
            (self.zz - self.cz)**2/self.rz**2 <= 1.0

        self.gray_values = np.ones((NZ, NY, NX), dtype=np.float64)
        # Gray value linealy varing in x direction
        # gray = np.linspace(0.1, 0.9, NX)

        if constant:
            self.gray_values = value * self.voxels
        else:
            zz_gray, yy_gray, xx_gray = np.meshgrid(
                np.linspace(0.0, 1.0, NZ),
                np.linspace(0.0, 1.0, NY),
                np.linspace(0.0, 1.0, NX), indexing="ij")

            self.gray_values = (zz_gray*0.3 + yy_gray*0.5 +
                                xx_gray*0.2) * self.voxels

        # self.gray_values = normalize(self.gray_values)
        # for x in range(NX):
            # self.gray_values[:, :, x] = gray[x] * self.voxels[:, :, x]


class Ellipse(Object):
    def __init__(self, cx, cy, rx, ry):
        super().__init__(cx, cy, 0)
        self.rx = rx
        self.ry = ry

    def create_voxels(self, grid):
        super().create_voxels(grid)
        NY, NX = grid.shape[:2]

        self.voxels = (self.xx - self.cx)**2/self.rx**2 + \
            (self.yy - self.cy)**2/self.ry**2 <= 1.0

        gray = np.linspace(0.1, 0.9, NX)
        self.gray_values = np.ones((NY, NX), dtype=np.float64)
        for x in range(NX):
            self.gray_values[:, x] = gray[x] * self.voxels[:, x]

        # for x in range(NX):
        #     for y in range(NY):
        #         if ~self.voxels[y, x]:
        #             self.gray_values[y, x] = 1.0
