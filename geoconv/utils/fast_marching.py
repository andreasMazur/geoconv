from matplotlib import pyplot as plt

import networkx as nx
import numpy as np

KEY_STATUS = "status"
KEY_ARRIVAL_TIME = "arrival_time"
KEY_VELOCITY = "velocity"

STATUS_ALIVE = "alive"
STATUS_NARROW = "narrow"
STATUS_FAR = "far"


def get_4_neighborhood(idx, grid_size):
    """Calculates the 4-neighborhood of the node referred to by 'idx'

    Parameters
    ----------
    idx: (int, int)
        The center node around which to calculate the 4 neighborhood.
    grid_size: int
        Size N for the NxN grid.

    Returns
    -------
    list:
        A list containing all valid neighbors of the node referred to by 'idx'
    """

    neighbors = [(idx[0] - 1, idx[1]),
                 (idx[0] + 1, idx[1]),
                 (idx[0], idx[1] - 1),
                 (idx[0], idx[1] + 1)]

    # Valid grid indices are greater than zero
    return [n for n in neighbors if 0 <= n[0] < grid_size and 0 <= n[1] < grid_size]


def determine_next_node(G):
    """Determines the node with the smallest arrival time in the narrow band.

    Parameters
    ----------
    G: nx.classes.graph.Graph
        The grid.

    Returns
    -------
    (int, int):
        The index of the node with the smallest arrival time in the narrowband
    """

    narrow_band = list({k: v for k, v in nx.get_node_attributes(G, KEY_STATUS).items() if v == STATUS_NARROW}.keys())
    try:
        closest_node = narrow_band[0]
    except IndexError:
        print()
    for idx in narrow_band[1:]:
        if G.nodes[idx][KEY_ARRIVAL_TIME] < G.nodes[closest_node][KEY_ARRIVAL_TIME]:
            closest_node = idx

    return closest_node


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
    for key in list(G.nodes):
        G.nodes[key][KEY_STATUS] = STATUS_FAR
        G.nodes[key][KEY_ARRIVAL_TIME] = np.inf
        G.nodes[key][KEY_VELOCITY] = wave_speed

    # Init start node
    G.nodes[start_node][KEY_STATUS] = STATUS_NARROW
    G.nodes[start_node][KEY_ARRIVAL_TIME] = 0.

    grid_attributes = nx.get_node_attributes(G, KEY_STATUS).values()
    while STATUS_NARROW in grid_attributes:
        closest_node_idx = determine_next_node(G)
        G.nodes[closest_node_idx][KEY_STATUS] = STATUS_ALIVE

        ########################################################
        # Recompute arrival times for neighbors of closest node
        ########################################################
        for selected in get_4_neighborhood(closest_node_idx, grid_size):
            if G.nodes[selected][KEY_STATUS] == STATUS_FAR:
                G.nodes[selected][KEY_STATUS] = STATUS_NARROW

            # Determine values to solve quadratic equation
            if G.nodes[selected][KEY_STATUS] != STATUS_ALIVE:
                selected_neighbors = get_4_neighborhood(selected, grid_size)
                x_neighbors = [t for t in selected_neighbors if t[0] == selected[0]]
                a = G.nodes[x_neighbors[0]][KEY_ARRIVAL_TIME]
                for n in x_neighbors[1:]:
                    a = G.nodes[n][KEY_ARRIVAL_TIME] if a > G.nodes[n][KEY_ARRIVAL_TIME] else a
                y_neighbors = [t for t in selected_neighbors if t[1] == selected[1]]
                b = G.nodes[y_neighbors[0]][KEY_ARRIVAL_TIME]
                for n in y_neighbors[1:]:
                    b = G.nodes[n][KEY_ARRIVAL_TIME] if b > G.nodes[n][KEY_ARRIVAL_TIME] else b

                # We assume that we have at least one neighbor with arrival time < np.inf
                if np.inf in [a, b]:
                    G.nodes[selected][KEY_ARRIVAL_TIME] = wave_speed + min(a, b)
                else:
                    G.nodes[selected][KEY_ARRIVAL_TIME] = (a + b) / 2 + np.sqrt(
                        (-a - b) ** 2 - (a ** 2 + b ** 2 - (wave_speed * grid_spacing) ** 2) / 2
                    )

        #########################################
        # pos = dict(G.nodes)
        # for key in pos.keys():
        #     pos[key] = key
        # nx.draw(G, pos, labels=nx.get_node_attributes(G, KEY_ARRIVAL_TIME), node_size=3000, node_color="white",
        #         edgecolors="black")
        # plt.show()
        #########################################
        grid_attributes = nx.get_node_attributes(G, KEY_STATUS).values()


if __name__ == "__main__":
    fast_marching(5)
