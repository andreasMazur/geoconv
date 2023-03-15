from matplotlib import pyplot as plt

import networkx as nx
import numpy as np

KEY_STATUS = "status"
KEY_ARRIVAL_TIME = "arrival_time"
KEY_VELOCITY = "velocity"

STATUS_ALIVE = "alive"
STATUS_NARROW = "narrow"
STATUS_FAR = "far"


def determine_next_node(G):
    """Determines the node with smallest arrival time in the narrow band.

    Parameters
    ----------
    G: nx.classes.graph.Graph
        The grid.

    Returns
    -------
    (int, int):
        The index of the node with the smallest arrival time in the narrow band
    """
    # TODO
    return 0, 0


def fast_marching(grid_size=5, wave_speed=1, start_node=(1, 1), grid_spacing=1):
    """Fast marching algorithm on orthogonal grids.

    Paper, that introduced the fast marching level set method:
    > [A fast marching level set method
       for monotonically advancing fronts](https://www.pnas.org/doi/abs/10.1073/pnas.93.4.1591)
    > James A. Sethian

    Further useful information on the fast marching algorithm:
    > [Fast Marching Methods for Computing
       Distance Maps and Shortest Paths](https://escholarship.org/content/qt7kx079v5/qt7kx079v5.pdf)
    > Ron Kimmel and James A. Sethian

    Parameters
    ----------
    grid_size: int
        The size N for the NxN 2D grid
    wave_speed: float
        The wave speed at the grid vertices (1 = compute Euclidean distance)
    start_node: (int, int)
        The index of the start node
    grid_spacing: float
        The x- and y- spacing between the nodes of the grid
    """
    G = nx.grid_2d_graph(grid_size, grid_size)

    #################
    # Initialization
    #################
    # {k:v for (k,v) in nx.get_node_attributes(G, "status").items() if v=="Alive"}
    # Stati: Alive, Narrow, Far
    # Node attributes: status, arrival_time, velocity
    for key in list(G.nodes):
        G.nodes[key][KEY_STATUS] = STATUS_FAR
        G.nodes[key][KEY_ARRIVAL_TIME] = np.inf
        G.nodes[key][KEY_VELOCITY] = wave_speed

    # Init start node
    G.nodes[start_node][KEY_STATUS] = STATUS_NARROW
    G.nodes[start_node][KEY_ARRIVAL_TIME] = 0.

    while True:  # TODO: termination condition
        closest_node_idx = determine_next_node(G)
        G.nodes[closest_node_idx][KEY_STATUS] = STATUS_ALIVE

        for neighbor_key in [(closest_node_idx[0] - 1, closest_node_idx[1]),
                             (closest_node_idx[0] + 1, closest_node_idx[1]),
                             (closest_node_idx[0], closest_node_idx[1] - 1),
                             (closest_node_idx[0], closest_node_idx[1] + 1)]:
            if neighbor_key[0] >= 0 and neighbor_key[1] >= 0:
                if G.nodes[neighbor_key][KEY_STATUS] != STATUS_ALIVE:
                    G.nodes[neighbor_key][KEY_STATUS] = STATUS_NARROW

                    ##########################
                    # Recompute arrival times - TODO: Simplify
                    ##########################
                    t_ij = G.nodes[neighbor_key][KEY_ARRIVAL_TIME]
                    mx = (neighbor_key[0] - 1, neighbor_key[1])  # TODO: Check whether node does not exceed boundaries
                    d_mx = (t_ij - G.nodes[mx][KEY_ARRIVAL_TIME]) / grid_spacing
                    px = (neighbor_key[0] + 1, neighbor_key[1])
                    d_px = (t_ij - G.nodes[px][KEY_ARRIVAL_TIME]) / grid_spacing
                    my = (neighbor_key[0], neighbor_key[1] - 1)
                    d_my = (t_ij - G.nodes[my][KEY_ARRIVAL_TIME]) / grid_spacing
                    py = (neighbor_key[0], neighbor_key[1] + 1)
                    d_py = (t_ij - G.nodes[py][KEY_ARRIVAL_TIME]) / grid_spacing

                    # TODO: Simplify case matching
                    d_x, d_y, f = max(d_mx, -d_px, 0), max(d_my, -d_py, 0), 1 / G.nodes[neighbor_key][KEY_VELOCITY]
                    if d_x > 0 and d_y > 0:
                        if d_mx >= -d_px:
                            b = G.nodes[mx][KEY_ARRIVAL_TIME]
                        else:
                            b = G.nodes[px][KEY_ARRIVAL_TIME]
                        if d_my >= -d_py:
                            d = G.nodes[my][KEY_ARRIVAL_TIME]
                        else:
                            d = G.nodes[py][KEY_ARRIVAL_TIME]
                        G.nodes[neighbor_key][KEY_ARRIVAL_TIME] = max(
                            ((b + d) / 2) + np.sqrt(((-b - d) / 2) ** 2 - (b ** 2 + d ** 2 - (G.nodes[neighbor_key][KEY_VELOCITY] * grid_spacing) ** 2) / 2),
                            ((b + d) / 2) - np.sqrt(((-b - d) / 2) ** 2 - (b ** 2 + d ** 2 - (G.nodes[neighbor_key][KEY_VELOCITY] * grid_spacing) ** 2) / 2)
                        )
                    elif d_x > 0:
                        if d_mx >= -d_px:
                            b = G.nodes[mx][KEY_ARRIVAL_TIME]
                        else:
                            b = G.nodes[px][KEY_ARRIVAL_TIME]
                        G.nodes[neighbor_key][KEY_ARRIVAL_TIME] = max(
                            b + np.sqrt(b ** 2 - (b ** 2 - (G.nodes[neighbor_key][KEY_VELOCITY] * grid_spacing) ** 2)),
                            b - np.sqrt(b ** 2 - (b ** 2 - (G.nodes[neighbor_key][KEY_VELOCITY] * grid_spacing) ** 2))
                        )
                    elif d_y > 0:
                        if d_my >= -d_py:
                            d = G.nodes[my][KEY_ARRIVAL_TIME]
                        else:
                            d = G.nodes[py][KEY_ARRIVAL_TIME]
                        G.nodes[neighbor_key][KEY_ARRIVAL_TIME] = max(
                            d + np.sqrt(d ** 2 - (d ** 2 - (G.nodes[neighbor_key][KEY_VELOCITY] * grid_spacing) ** 2)),
                            d - np.sqrt(d ** 2 - (d ** 2 - (G.nodes[neighbor_key][KEY_VELOCITY] * grid_spacing) ** 2))
                        )
                    else:  # a and b <= 0
                        G.nodes[neighbor_key][KEY_ARRIVAL_TIME] = (1 / G.nodes[neighbor_key][KEY_VELOCITY]) + min(
                            min(G.nodes[mx][KEY_ARRIVAL_TIME], G.nodes[px][KEY_ARRIVAL_TIME]),
                            min(G.nodes[my][KEY_ARRIVAL_TIME], G.nodes[py][KEY_ARRIVAL_TIME])
                        )

    pos = dict(G.nodes)
    for key in pos.keys():
        pos[key] = key
    nx.draw(G, pos, labels=nx.get_node_attributes(G, "status"), node_size=3000, node_color="white", edgecolors="black")
    plt.show()


if __name__ == "__main__":
    fast_marching(5)
