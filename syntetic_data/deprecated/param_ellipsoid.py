from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from transforms import roty


# rx, ry, rz = 1, 2, 2
# cx, cy, cz = 2, 3, 1

# # Set of all sphereical angles
# u = np.linspace(0, 2*np.pi, 100)
# v = np.linspace(0, np.pi, 100)

# # Cartesian coordinates that correspond to the spherical angles
# x = (rx * np.outer(np.cos(u), np.sin(v))) + cx
# y = (ry * np.outer(np.sin(u), np.sin(v))) + cy
# z = (rz * np.outer(np.ones_like(u), np.cos(v))) + cz

# fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, rstride=4, cstride=4,  color='b', alpha=0.4)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # print(x)
# # print(x.shape, y.shape, z.shape)

# t = np.array([0, 1, 0])
# R = roty(90)

# coords = np.stack((x, y, z), axis=2)

# new_coords = np.zeros((100, 100, 3))


# print(coords.shape, new_coords.shape)
# for i in range(100):
#     for j in range(100):
#         p = coords[i, j]
#         new_coords[i, j] = R@p

# ax.plot_surface(new_coords[:, :, 0], new_coords[:, :, 1],
#                 new_coords[:, :, 2], rstride=4, cstride=4,  color='r', alpha=0.4)

# # Adjustment of the axes, so that they all have the same span:

# # for axis in 'xyz':
# #     getattr(ax, 'set_{}lim'.format(axis))((-5, 5))

# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# grid points in spherical coord:
N = 80
R = 2
x, y, z = np.meshgrid(np.linspace(-R, R, N),
                      np.linspace(-R, R, N), np.linspace(-R, R, N))

# filter points outside ellipsoid interior:
mask = (2*x)**2 + (3*y)**2 + z**2 <= R**2
x = x[mask]
y = y[mask]
z = z[mask]


# convert to cartesian for plotting:

ax.scatter3D(x, y, z)
# ax.set_xlim(-1.2,1.2)
# ax.set_ylim(-1.2,1.2)
# ax.set_zlim(-1.2,1.2)
plt.show()
