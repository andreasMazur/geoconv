from matplotlib import pyplot as plt

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