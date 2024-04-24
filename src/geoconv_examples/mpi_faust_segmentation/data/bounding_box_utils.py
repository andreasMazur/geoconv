import numpy as np
import trimesh


def visualize_bbs(mesh, bbs, verbose=False):
    """Visualizes bounding boxes.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh that shall be segmented.
    bbs: list
        A list of bounding boxes.
    verbose: bool
        Whether to plot the bounding boxes

    Returns
    -------
    np.ndarray:
        Vertex indices which are within the bounding boxes.
    """
    # Initialize vertices and their colors
    vertices = np.array(mesh.vertices)
    colors = [[0, 0, 0, 255] for _ in range(vertices.shape[0])]
    are_within = []

    for bb in bbs:
        # Get bounding box corner points
        corners = bb.corners()

        # Update vertex colors depending on whether they are inside one bounding box
        for idx, vertex in enumerate(mesh.vertices):
            if bb.is_within(vertex):
                colors[idx] = [0, 255, 0, 255]
                are_within.append(idx)

        # Add corner points to plot
        vertices = np.concatenate([vertices, corners])

        # Add bounding box to colors
        colors = colors + [[255, 0, 0, 255] for _ in range(8)]

    # Visualize point cloud
    if verbose:
        pc = trimesh.PointCloud(vertices=vertices, colors=colors)
        pc.show()

    return np.unique(np.array(are_within))
