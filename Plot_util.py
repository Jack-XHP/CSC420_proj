from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def plot_frsutum(box3d_corner, point_2d, velo_in_box_fov):
    box3d_corner_show = box3d_corner[:, [2, 0, 1]]
    point_2d_show = point_2d[:, [2, 0, 1]]
    velo_in_box_fov_show = velo_in_box_fov[:, [2, 0, 1]]
    edges = [
        [box3d_corner_show[0], box3d_corner_show[1], box3d_corner_show[2], box3d_corner_show[3]],
        [box3d_corner_show[4], box3d_corner_show[5], box3d_corner_show[6], box3d_corner_show[7]],
        [box3d_corner_show[0], box3d_corner_show[1], box3d_corner_show[5], box3d_corner_show[4]],
        [box3d_corner_show[2], box3d_corner_show[3], box3d_corner_show[7], box3d_corner_show[6]],
        [box3d_corner_show[1], box3d_corner_show[2], box3d_corner_show[6], box3d_corner_show[5]],
        [box3d_corner_show[4], box3d_corner_show[7], box3d_corner_show[3], box3d_corner_show[0]],
        [box3d_corner_show[2], box3d_corner_show[3], box3d_corner_show[7], box3d_corner_show[6]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0, 0, 1, 0.1))

    mask = point_2d[:, 2].flatten() < 50
    points_t = point_2d_show[mask, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.add_collection3d(faces)
    ax.scatter(points_t[:, 0].flatten(), points_t[:, 1].flatten(), points_t[:, 2].flatten(), c='r',
               marker='o')
    ax.scatter(velo_in_box_fov_show[:, 0].flatten(), velo_in_box_fov_show[:, 1].flatten(),
               velo_in_box_fov_show[:, 2].flatten(), c='b',
               marker='^')
    ax.invert_zaxis()
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    set_axes_equal(ax)
    plt.show()


def plot_all(points, image_y, image_x, calib, velo_rect_t):
    points = points.reshape(-1, 3)
    points[points[:, 2] > 1000] = np.zeros(3)
    points = points.reshape(image_y, image_x, 3)
    pointsk = points[0, 0:600].reshape(-1, 3)
    pointsk = calib.project_rect_to_image(pointsk)
    plt.scatter(pointsk[:, 0], pointsk[:, 1])
    plt.show()

    p = [(points[:150, :600].reshape(-1, 3), 'r'), (points[150:, :600].reshape(-1, 3), 'b'),
         (points[:150, 600:].reshape(-1, 3), 'g'), (points[150:, 600:].reshape(-1, 3), 'm')]
    velo_rect_t = velo_rect_t[:, [2, 0, 1]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    for k in p:
        points_t = k[0]
        print(points_t.shape)
        c = k[1]
        choice = np.random.choice(points_t.shape[0], 1500, replace=True)
        points_t = points_t[choice]
        points_t = points_t[:, [2, 0, 1]]
        ax.scatter(points_t[:, 0].flatten(), points_t[:, 1].flatten(), points_t[:, 2].flatten(), c=c, marker='o')
    ax.scatter(velo_rect_t[:, 0].flatten(), velo_rect_t[:, 1].flatten(), velo_rect_t[:, 2].flatten(), c='y', marker='^')
    ax.invert_zaxis()
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    set_axes_equal(ax)
    plt.show()
