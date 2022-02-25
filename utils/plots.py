import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def plot_slices(X, str="", block=True):
    """
    X:  numpy array with shape [X,Y,Z]
    """
    fig, _ = plt.subplots(3, X.shape[2] // 3)
    plt.suptitle(f"{str} {X.shape}")
    for i, ax in enumerate(fig.get_axes()):
        ax.imshow(X[:, :, i].T, cmap="gray", origin="lower")
        # ax.imshow(X[:, :, i], cmap="gray")
    plt.show(block=block)


def plot_slice(slice, block=True):
    plt.figure()
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.title(f"{slice.shape}")
    plt.show(block=block)


def plot_3d(image, threshold=-300):
    # p = image.transpose(2,1,0)
    p = image
    verts, faces, normals, values = measure.marching_cubes(
        p, 25.0, method="lewiner")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
