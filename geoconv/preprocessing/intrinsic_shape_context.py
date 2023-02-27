from matplotlib import pyplot as plt

import numpy as np
import trimesh


def init_surface_bins(object_mesh, y, n_rays=10):
    """Computes angular coordinates for the surface bins.

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        The object mesh on which to calculate the local polar coordinate system
    y: int
        The index of the center vertex around which the local polar coordinate system will be calculated
    n_rays: int
        Amount of rays to use for segmentation

    Returns
    -------
    np.ndarray:
        Angular surface coordinates for the ISC-binning procedure
    """
    # Calculate equi-distant angular rays
    angle_width = 2 * np.pi / n_rays
    plane_segment_borders = np.array([a * angle_width for a in range(n_rays)])

    # Find all triangles of y
    indices = np.where(object_mesh.triangles == object_mesh.vertices[y])[0]
    uniques, counts = np.unique(indices, return_counts=True)
    indices = uniques[counts == 3]
    tri = object_mesh.triangles[indices]

    # Compute angles at y (considering found triangles)
    indices = np.unique(np.where(tri == object_mesh.vertices[y])[1].reshape((-1, 3)), axis=1).reshape(-1)
    surface_angles = trimesh.triangles.angles(tri)[(range(tri.shape[0]), indices)]

    # Map segment borders onto surface (angle ratios on plane = angle ratios on surface)
    surface_segment_borders = plane_segment_borders / (2 * np.pi) * surface_angles.sum()

    return surface_segment_borders


def unfolding_procedure():
    """Propagate a path through multiple adjacent triangles across a triangle mesh

    Paper, that introduced intrinsic shape context descriptors:
    > [Computing Geodesic Paths on Manifolds](https://www.pnas.org/doi/abs/10.1073/pnas.95.15.8431)
    > Ron Kimmel and James A. Sethian

    Parameters
    ----------

    Returns
    -------

    """
    pass


def compute_isc():
    """

    Paper, that introduced intrinsic shape context descriptors:
    > [Intrinsic Shape Context Descriptors for Deformable Shapes](https://ieeexplore.ieee.org/document/6247671)
    > Iasonas Kokkinos, Michael M. Bronstein, Roee Litman and Alex M. Bronstein

    Parameters
    ----------

    Returns
    -------

    """

if __name__ == "__main__":
    faust_dir = "/home/andreas/Uni/Masterarbeit/MPI-FAUST/training/registrations"
    ref_mesh = f"{faust_dir}/tr_reg_000.ply"
    mesh = trimesh.load_mesh(ref_mesh)

    init_surface_bins(mesh, 0)
