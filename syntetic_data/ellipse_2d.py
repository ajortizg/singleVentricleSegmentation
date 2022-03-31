import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)


class Ellipse:
    def __init__(self, cx, cy, rx, ry, angle):
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        self.angle = angle

        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        self.x = rx * np.cos(theta)
        self.y = ry * np.sin(theta)
        rad_ang = np.radians(angle)
        R = np.array([
            [np.cos(rad_ang), -np.sin(rad_ang)],
            [np.sin(rad_ang), np.cos(rad_ang)],
        ])

        self.x, self.y = np.dot(R, np.array([self.x, self.y]))
        self.x += cx
        self.y += cy
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
        return f"cx: {self.cx}, cy: {self.cy}, rx: {self.rx}, ry: {self.ry}, angle: {self.angle}"


eA = Ellipse(cx=0, cy=0, rx=10, ry=20, angle=45)
eB = Ellipse(cx=3, cy=4, rx=5, ry=25, angle=90)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto')
ax.fill(eA.x, eA.y, alpha=0.2, facecolor='yellow',
        edgecolor='black', linewidth=2, zorder=1)

ax.fill(eB.x, eB.y, alpha=0.2, facecolor='red',
        edgecolor='black', linewidth=2, zorder=1)
ax.set_xlabel("X")
ax.set_ylabel("Y")

ts = 10
alpha = np.linspace(0.0, 1.0, 10)
dif = eB - eA
for i in range(ts):
    ei = eA*(1-alpha[i]) + eB*(alpha[i])  # fwd (A -> B)
    # ei = eA*(alpha[i]) + eB*(1.0-alpha[i])  # bwd (B -> A)
    print(ei)

print("\neA: ", eA)
print("\neB: ", eB)

plt.show()


# NY, NX = RY*3, RX*3  # Image height and widht
# CY, CX = NY // 2, NX // 2  # Image center coordinates

# print(NY, NX)
# print(CY, CX)

# img = np.ones((NY, NX))

# for i in range(x.shape[0]):
#     xi = int(x[i] + CX)
#     yi = int(y[i] + CY)
#     # if (xi**2/RX**2) + (yi*2/RY**2) <= 1.0:
#     img[yi, xi] = 0.0
#     # print(xi**2/RX**2, yi**2/RY**2)

# # for i in range(NY):
# #     for j in range(NX):


# fig = plt.figure()
# plt.imshow(img, cmap="gray", origin="upper")


# for i in range(NY):
#     for j in range(NX):
#         img[i, j] = 1


# self.voxels = (self.xx - self.cx) ** 2 / self.rx**2 + (self.yy - self.cy) ** 2 / self.ry**2 <= 1.0


# print(img.shape)
