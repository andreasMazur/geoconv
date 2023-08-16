from geoconv.utils.misc import get_included_faces, normalize_mesh

from matplotlib import pyplot as plt

import pygeodesic.geodesic as geodesic
import trimesh
import numpy as np
import sys


def princeton_benchmark(imcnn, test_dataset, ref_mesh_path, file_name):
    """Plots the accuracy w.r.t. a gradually changing geodesic error

    Princeton benchmark has been introduced in:
    > [Blended intrinsic maps](https://doi.org/10.1145/2010324.1964974)
    > Vladimir G. Kim, Yaron Lipman and Thomas Funkhouser

    Parameters
    ----------
    imcnn: tf.keras.Model
        The Intrinsic Mesh CNN
    test_dataset: tf.data.Dataset
        The test dataset on which to evaluate the Intrinsic Mesh CNN
    ref_mesh_path: str
        A path to the reference mesh
    file_name: str
        The file name under which to store the plot and the data (without file format ending!)
    """
    reference_mesh = trimesh.load_mesh(ref_mesh_path)
    reference_mesh = normalize_mesh(reference_mesh)
    geoalg = geodesic.PyGeodesicAlgorithmExact(reference_mesh.vertices, reference_mesh.faces)

    geodesic_errors, mesh_idx = [], -1
    for ((signal, barycentric), ground_truth) in test_dataset:
        mesh_idx += 1
        prediction = imcnn([signal, barycentric]).numpy().argmax(axis=1)
        pred_idx = -1
        for gt, pred in np.stack([ground_truth, prediction], axis=1):
            pred_idx += 1
            sys.stdout.write(f"\rCurrently at mesh {mesh_idx} - Prediction {pred_idx}")
            geodesic_errors.append(geoalg.geodesicDistance(pred, gt)[0])
    geodesic_errors = np.array(geodesic_errors)
    geodesic_errors.sort()

    ###########
    # Plotting
    ###########
    amt_values = geodesic_errors.shape[0]
    arr = np.array([((i + 1) / amt_values, x) for (i, x) in zip(range(amt_values), geodesic_errors)])
    plt.plot(arr[:, 1], arr[:, 0])
    plt.title("Princeton Benchmark")
    plt.xlabel("geodesic error")
    plt.ylabel("% correct correspondences")
    plt.grid()
    plt.savefig(f"{file_name}.svg")
    np.save(f"{file_name}.npy", arr)
    plt.show()


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
