from matplotlib import pyplot as plt

import networkx as nx
import numpy as np


def fast_marching(grid_size=5):
    """Fast marching algorithm on orthogonal grids.

    Paper, that introduced the fast marching level set method:
    > [A fast marching level set method
       for monotonically advancing fronts](https://www.pnas.org/doi/abs/10.1073/pnas.93.4.1591)
    > James A. Sethian

    Parameters
    ----------
    grid_size: int
        The size N for the NxN 2D grid

    Returns
    -------

    """
    G = nx.grid_2d_graph(grid_size, grid_size)

    for key in list(G.nodes):
        G.nodes[key]["status"] = "Far Away"
        G.nodes[key]["time"] = np.inf
        G.nodes[key]["speed"] = np.random.uniform()  # random positive speeds at each grid point

    for i in range(grid_size):
        G.nodes[(0, i)]["status"] = "Alive"
        G.nodes[(0, i)]["time"] = 0.
        G.nodes[(1, i)]["status"] = "Narrow"
        G.nodes[(1, i)]["time"] = 0. / G.nodes[(1, i)]["speed"]  # TODO: initial times for narrow points

    pos = dict(G.nodes)
    for key in pos.keys():
        pos[key] = key
    nx.draw(G, pos, labels=nx.get_node_attributes(G, "status"), node_size=3000, node_color="white", edgecolors="black")
    plt.show()


if __name__ == "__main__":
    fast_marching(5)
