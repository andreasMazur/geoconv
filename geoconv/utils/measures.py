from geoconv.utils.misc import get_included_faces

import numpy as np


def kernel_coverage(object_mesh, gpc_system, bary_coordinates):
    """Quality measure for how much a kernel covers within a GPC-system

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        An object mesh
    gpc_system:
        The considered GPC-system
    bary_coordinates:
        The barycentric coordinates for the kernel for the considered GPC-system


    Returns
    -------
    float
        The ratio between faces in which a kernel vertex lies and the total
        amount of faces within the GPC-system (kernel coverage)
    """
    patch = get_included_faces(object_mesh, gpc_system)
    patch_faces = object_mesh.faces[patch]
    covered_faces = []
    for angular in range(bary_coordinates.shape[0]):
        for radial in range(bary_coordinates.shape[1]):
            face_of_kernel_vertex = bary_coordinates[angular, radial, :, 0]

            # Check in what face the kernel vertex lies
            arr = np.array(face_of_kernel_vertex == patch_faces)
            arr = np.logical_and(np.logical_and(arr[:, 0], arr[:, 1]), arr[:, 2])
            arr = np.where(arr)[0]
            if len(arr):
                # Add face index to known faces if previously unknown
                face_idx = patch[arr[0]]
                if face_idx not in covered_faces:
                    covered_faces.append(face_idx)

    # Return the percentage of how many triangles in the patch are regarded by the kernel
    return len(covered_faces) / len(patch)


def evaluate_kernel_coverage(object_mesh, gpc_systems, bary_coordinates, verbose=True):
    """Computes the average kernel coverage of a GPC-system

    Parameters
    ----------
    object_mesh: trimesh.Trimesh
        An object mesh
    gpc_systems: np.ndarray
        A 3D-array containing multiple GPC-systems
    bary_coordinates:
        The barycentric coordinates of the kernels layed on the given GPC-systems

    Returns
    -------
    float
        The average coverage rate of the kernels on the GPC-systems
    """
    assert gpc_systems.shape[0] == bary_coordinates.shape[0], \
        "You must provide similar amount of GPC-system and Barycentric coordinate arrays."

    coverages = []
    for idx in range(gpc_systems.shape[0]):
        coverage = kernel_coverage(object_mesh, gpc_systems[idx], bary_coordinates[idx])
        coverages.append(coverage)
    if verbose:
        print(coverages)
    return np.mean(coverages)
