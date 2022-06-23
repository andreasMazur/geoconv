from geodesic_polar_coordinates.discrete_gpc import local_gpc
from matplotlib import pyplot as plt

import os
import trimesh
import numpy as np
import matplotlib


matplotlib.use('TkAgg')


def visualize_linear_combination(vertex_i, vertex_j, vertex_k, vertex_i_idx, vertex_j_idx, vertex_k_idx, s):
    """Visualizes linear combination for computing vertex s.

    This function shall illustrate what Figure 6 illustrates in [*].

    [*]:
    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    **Input**

    - vertex_i
    - vertex_j
    - vertex_k
    - vertex_i_idx
    - vertex_j_idx
    - vertex_k_idx
    - s

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data = np.array([vertex_i, vertex_j, vertex_k, vertex_i])
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.scatter(s[0], s[1], s[2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    vertex_name = ["vertex_i", "vertex_j", "vertex_k"]
    for i, label in enumerate([vertex_i_idx, vertex_j_idx, vertex_k_idx]):
        ax.text(data[i, 0], data[i, 1], data[i, 2], vertex_name[i])
    ax.text(s[0], s[1], s[2], "s")

    for v in [vertex_i, vertex_j, vertex_k]:
        temp = np.array([s, v])
        ax.plot(temp[:, 0], temp[:, 1], temp[:, 2], color="r")

    plt.show()


def visualize_linear_combination(vertex_i, vertex_j, vertex_k, vertex_i_idx, vertex_j_idx, vertex_k_idx, s):
    """Visualizes linear combination for computing vertex s.

    This function shall illustrate what Figure 6 illustrates in [*].

    [*]:
    > [Geodesic polar coordinates on polygonal
    meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03187.x)
    > Melvær, Eivind Lyche, and Martin Reimers.

    **Input**

    - vertex_i
    - vertex_j
    - vertex_k
    - vertex_i_idx
    - vertex_j_idx
    - vertex_k_idx
    - s

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data = np.array([vertex_i, vertex_j, vertex_k, vertex_i])
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.scatter(s[0], s[1], s[2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    vertex_name = ["vertex_i", "vertex_j", "vertex_k"]
    for i, label in enumerate([vertex_i_idx, vertex_j_idx, vertex_k_idx]):
        ax.text(data[i, 0], data[i, 1], data[i, 2], vertex_name[i])
    ax.text(s[0], s[1], s[2], "s")

    for v in [vertex_i, vertex_j, vertex_k]:
        temp = np.array([s, v])
        ax.plot(temp[:, 0], temp[:, 1], temp[:, 2], color="r")

    plt.show()


def visualize_gpc(source_point, radial_coordinates, angular_coordinates, object_mesh):
    """Visualizes the radial and angular coordinates of a local GPC-system on an object mesh.

    **Input**

    - The local GPC-system to visualize.
    - The corresponding object mesh.

    """

    # Visualize radial coordinates
    radial_coordinates[radial_coordinates == np.inf] = 0.0
    colors = trimesh.visual.interpolate(radial_coordinates, color_map="Reds")
    colors[source_point] = np.array([255, 255, 0, 255])
    point_cloud = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    point_cloud.show()

    # Visualize angular coordinates
    colors = trimesh.visual.interpolate(angular_coordinates, color_map="YlGn")
    colors[source_point] = np.array([255, 255, 0, 255])
    # for idx in range(angular_coordinates.shape[0]):
    #     if angular_coordinates[idx] == 0.0:
    #         colors[idx] = np.array([255, 255, 0, 255])
    point_cloud = trimesh.points.PointCloud(object_mesh.vertices, colors=colors)
    point_cloud.show()


if __name__ == "__main__":
    faust_dir = "/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations"
    file_list = os.listdir(faust_dir)
    file_list.sort()
    file_list = [f for f in file_list if f[-4:] != ".png"]

    # Choose mesh here
    chosen_file = file_list[0]
    object_mesh_ = trimesh.load_mesh(f"{faust_dir}/{chosen_file}")

    # Source point examples: 10, 920, 930, 1200, 1280, 1330, 2000, 2380
    source_point_ = 2380
    u, theta, _, _ = local_gpc(source_point_, u_max=0.05, object_mesh=object_mesh_, use_c=True, eps=0.000001)
    visualize_gpc(source_point_, u, theta, object_mesh_)
