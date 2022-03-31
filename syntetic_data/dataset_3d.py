import matplotlib.pyplot as plt
import numpy as np
from ellipse import Ellipsoid
from transforms import rotx, roty, rotz

eA = Ellipsoid(0, 0, 0, 3, 4, 5, 0, 15, 0)
eB = Ellipsoid(0, 13, 0, 1, 2, 4, 90, 45, 33)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='auto')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("ellipsoids")
ax.plot_surface(eA.x, eA.y, eA.z, color='green', alpha=0.3, linewidth=2)
ax.plot_surface(eB.x, eB.y, eB.z, color='red', alpha=0.3, linewidth=2)

# Compute intermediate steps
ts = 5
alpha = np.linspace(0.0, 1.0, ts)
prev_ei = None
ellipsoids = []
flos = []
for i in range(ts):
    ei = eA*(1-alpha[i]) + eB*(alpha[i])  # fwd (A -> B)
    # ei = eA*(alpha[i]) + eB*(1.0-alpha[i])  # bwd (B -> A)
    # print(ei)
    ax.plot_surface(ei.x, ei.y, ei.z, color='blue', alpha=0.1, linewidth=2)

    # compute fwd optical flow
    if i >= 1:
        u = ei.x - prev_ei.x
        v = ei.y - prev_ei.y
        w = ei.z - prev_ei.z
        ellipsoids.append((prev_ei, ei))
        flos.append(np.array([u, v, w]))

    prev_ei = ei


# Warping
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='auto')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("optical flow")
for i, ((prev, ei), flo) in enumerate(zip(ellipsoids, flos)):
    ax.plot_surface(prev.x, prev.y, prev.z,
                    color='green', alpha=0.3, linewidth=2)

    ax.plot_surface(prev.x + flo[0, :, :], prev.y + flo[1, :, :],
                    prev.z+flo[2, :, :], color='red', alpha=0.3, linewidth=2)

    print(f"\npair {i}\n{prev}")
    print(ei)

print(f"\neA: {eA}")
print(f"eB: {eB}")


# for axis in 'xyz':
#     getattr(ax, 'set_{}lim'.format(axis))((-8, 8))

plt.show()
