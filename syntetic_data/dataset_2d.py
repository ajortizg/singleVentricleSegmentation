import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import time
from ellipse import Ellipse

np.set_printoptions(precision=2, suppress=True)


# Create initial and final ellipsoids
eA = Ellipse(cx=10, cy=10, rx=10, ry=20, angle=45)
eB = Ellipse(cx=3, cy=4, rx=5, ry=25, angle=90)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto')
ax.fill(eA.x, eA.y, alpha=0.4, facecolor='green',
        edgecolor='black', linewidth=2, zorder=1)

ax.fill(eB.x, eB.y, alpha=0.4, facecolor='red',
        edgecolor='black', linewidth=2, zorder=1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("ellipses")

# Compute intermediate steps
ts = 10
alpha = np.linspace(0.0, 1.0, ts)
prev_ei = None
ellipsoids = []
flos = []
for i in range(ts):
    ei = eA*(1-alpha[i]) + eB*(alpha[i])  # fwd (A -> B)
    # ei = eA*(alpha[i]) + eB*(1.0-alpha[i])  # bwd (B -> A)
    # print(ei)
    ax.fill(ei.x, ei.y, alpha=0.1, facecolor='blue',
            edgecolor='black', linewidth=2, zorder=1)

    # compute fwd optical flow
    if i >= 1:
        u = ei.x - prev_ei.x
        v = ei.y - prev_ei.y
        ellipsoids.append((prev_ei, ei))
        flos.append(np.array([u, v]))

    prev_ei = ei


print("\neA: ", eA)
print("eB: ", eB)


# Warping
fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("optical flow")
for i, ((prev, ei), flo) in enumerate(zip(ellipsoids, flos)):
    ax.fill(prev.x, prev.y, alpha=0.1, facecolor='green',
            edgecolor='black', linewidth=2, zorder=1)

    ax.fill(prev.x + flo[0, :], prev.y + flo[1, :], alpha=0.1, facecolor='red',
            edgecolor='black', linewidth=2, zorder=1)
    print(f"\npair {i}\n{prev}")
    print(ei)

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
