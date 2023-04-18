import matplotlib.pyplot as plt
import numpy as np
from ellipse import Ellipsoid
from transforms import rotx, roty, rotz
import utilities
from tqdm import trange
import sys

np.set_printoptions(precision=2, suppress=True)


def plot_ellipsoids(ax, ellipsoids):
    for ei in ellipsoids:
        ax.plot_surface(ei.x, ei.y, ei.z,
                        color=ei.color, alpha=0.3, linewidth=2)


def combine_voxels(ellipsoids):
    NZ, NY, NX = ellipsoids[0].mask.shape
    img = np.zeros((NZ, NY, NX))

    e1 = ellipsoids[0]
    e2 = ellipsoids[1]
    e3 = ellipsoids[2]

    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                if e3.mask[z, y, x]:
                    img[z, y, x] = e3.voxels[z, y, x]
                elif e2.mask[z, y, x]:
                    img[z, y, x] = e2.voxels[z, y, x]
                elif e1.mask[z, y, x]:
                    img[z, y, x] = e1.voxels[z, y, x]
                else:
                    img[z, y, x] = 0.05
    return img


# Size of voxel map
NZ, NY, NX = 16, 352, 352
grid = utilities.create_grid(NZ, NY, NX)

e1A = Ellipsoid(cx=0, cy=0, cz=0, rx=90, ry=150, rz=9, angx=0, angy=0, angz=0)
e1A.create_voxels(grid, False, value1=0.1, value2=0.3)
e2A = Ellipsoid(cx=0, cy=0, cz=0, rx=50, ry=100, rz=9, angx=0, angy=0, angz=0)
e2A.create_voxels(grid, False, value1=0.3, value2=0.6)
e3A = Ellipsoid(cx=0, cy=0, cz=0, rx=30, ry=50, rz=9, angx=0, angy=0, angz=0)
e3A.create_voxels(grid, False, value1=0.6, value2=0.9)
eAs = [e1A, e2A, e3A]


e1B = Ellipsoid(cx=10, cy=-10, cz=0, rx=90, ry=150,
                rz=9, angx=0, angy=0, angz=-10)
e1B.create_voxels(grid, False, value1=0.1, value2=0.3)
e2B = Ellipsoid(cx=10, cy=-18, cz=0, rx=50, ry=100,
                rz=9, angx=0, angy=0, angz=15)
e2B.create_voxels(grid, False, value1=0.3, value2=0.6)
e3B = Ellipsoid(cx=15, cy=-10, cz=0, rx=30, ry=50,
                rz=9, angx=0, angy=0, angz=12)
e3B.create_voxels(grid, False, value1=0.6, value2=0.9)
eBs = [e1B, e2B, e3B]


imgA = combine_voxels(eAs)
imgB = combine_voxels(eBs)


# Compute intermediate steps
ts = 10
alpha = np.linspace(0.0, 1.0, ts)
# ellipsoids = []

for i in trange(ts):
    es = []
    for eA, eB in zip(eAs, eBs):
        ei = eA*(1-alpha[i]) + eB*(alpha[i])  # fwd (A -> B)
        ei.create_voxels(grid,  eA.constant, eA.value1, eA.value2)
#         # ei = eA*(alpha[i]) + eB*(1.0-alpha[i])  # bwd (B -> A)
        # print(ei)
        es.append(ei)

    img_t = combine_voxels(es)
    np.save(f"data/Synthetic3D/NIFTI_4D_Datasets/vol_{i}.npy", img_t)
    np.save(
        f"data/Synthetic3D/NIFTI_Single_Ventricle_Segmentations/mask_{i}.npy", es[-1].mask)

    # ellipsoids.append(es)

# t = 9
# e = ellipsoids[9]


# img_t0 = combine_voxels(e)
utilities.plot_slices(imgA, str="A", block=False)
utilities.plot_slices(imgB, str="B", block=True)
# utils.plot_slices(img_t0, str="t0", block=True)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d', aspect='auto')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("ellipsoids")
# plot_ellipsoids(ax, eAs)
