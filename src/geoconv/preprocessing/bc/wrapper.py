from geoconv.preprocessing.bc.bc_utils import create_template_matrix, interpolation

from tqdm import tqdm
from multiprocessing import Pool

import numpy as np


def compute_barycentric_coordinates(atlas, n_radial=2, n_angular=4, radius=0.05, processes=1):
    """Compute the barycentric coordinates for a given atlas.

    Parameters
    ----------
    atlas: Atlas
        The atlas for the underlying mesh.
    n_radial: int
        The amount of radial coordinates of the template you wish to use.
    n_angular: int
        The amount of angular coordinates of the template you wish to use.
    radius: float
        The radius of the template of the template you wish to use.
    processes: int
        The amount of processes to use for parallel computation.

    Returns
    -------
    A 5D-array containing the Barycentric coordinates for each template vertex and each chart. It has the following
    structure:
        B[a, b, c, d, e]:
            - a: References chart centered in vertex `a` of object mesh `object_mesh`
            - b: References the b-th radial coordinate of the template
            - c: References the c-th angular coordinate of the template
            - B[a, b, c, :, 0]: Returns the **indices** of the nodes that construct the triangle containing the template
                                vertex (b, c) in chart centered in node `a`
            - B[a, b, c, :, 1]: Returns the **barycentric coordinates** of the nodes that construct the triangle
                                containing the template vertex (b, c) in chart centered in node `a`
    """
    # Define template vertices at which interpolation values will be needed
    template_matrix = create_template_matrix(
        n_radial=n_radial, n_angular=n_angular, radius=radius, in_cart=True
    )

    chart_indices = range(atlas.triangle_mesh.vertices.shape[0]) if atlas.chart_indices is None else atlas.chart_indices
    n_charts = len(chart_indices)

    triples = [
       (
           template_matrix[radial_coordinate, angular_coordinate],
           atlas.chart_triangles[chart_idx],
           atlas.chart_faces[chart_idx]
       )
        for chart_idx in chart_indices
        for angular_coordinate in range(n_angular)
        for radial_coordinate in range(n_radial)
    ]
    with Pool(processes) as p:
        bc = p.starmap(
            interpolation,
            tqdm(triples, total=len(triples), postfix="Computing barycentric coordinates.."),
        )

    return np.array(bc).reshape(n_charts, n_radial, n_angular, 3, 2)
