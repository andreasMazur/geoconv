from geoconv.utils.misc import normalize_mesh

from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

import pygeodesic.geodesic as geodesic
import trimesh
import numpy as np


def princeton_benchmark(imcnn,
                        test_dataset,
                        ref_mesh_path,
                        file_name,
                        normalize=True,
                        plot_title="Princeton Benchmark",
                        curve_label=None,
                        plot=True,
                        processes=1,
                        geodesic_diameter=None,
                        pytorch_model=False):
    """Plots the accuracy w.r.t. a gradually changing geodesic error

    Princeton benchmark has been introduced in:
    > [Blended intrinsic maps](https://doi.org/10.1145/2010324.1964974)
    > Vladimir G. Kim, Yaron Lipman and Thomas Funkhouser

    Parameters
    ----------
    imcnn:
        The Intrinsic Mesh CNN. If it has multiple outputs, then this function expects the vertex-classifications
        to be the first returned tensor.
    test_dataset: tensorflow.data.Dataset
        The test dataset on which to evaluate the Intrinsic Mesh CNN
    ref_mesh_path: str
        A path to the reference mesh
    file_name: str
        The file name under which to store the plot and the data (without file format ending!)
    normalize: bool
        Whether to normalize the reference mesh
    plot_title: str
        The title of the plot
    curve_label: str
        The name displayed in the plot legend
    plot: bool
        Whether to plot immediately.
    processes: int
        The amount of concurrent processes.
    geodesic_diameter: float
        The geodesic diameter of the reference mesh
    pytorch_model: bool
        Whether a pytorch model is given.
    """

    reference_mesh = trimesh.load_mesh(ref_mesh_path)
    if normalize:
        reference_mesh, _ = normalize_mesh(reference_mesh, geodesic_diameter=geodesic_diameter)

    mesh_number = 0
    for ((signal, barycentric), ground_truth) in test_dataset:
        # Get predictions of the model
        if pytorch_model:
            prediction = imcnn([signal, barycentric]).cpu()
            ground_truth = ground_truth.cpu()
        else:
            prediction = imcnn([signal, barycentric])

        # Handle situation in which model returns multiple outputs
        if isinstance(prediction, tuple):
            prediction = np.array(prediction).argmax(axis=-1)
        else:
            prediction = np.array(prediction[0]).argmax(axis=-1)

        # Create ground-truth/prediction-pairs and prepare data for multiprocessing
        # TODO: Account for batch sizes > 1!  'np.stack([ground_truth, prediction], axis=-1)-->[0]<--'
        batched = [(data, reference_mesh) for data in np.stack([ground_truth, prediction], axis=-1)[0]]

        # Calculate geodesic distance of ground-truth to prediction on the given reference mesh
        with Pool(processes) as p:
            geodesic_errors = p.starmap(
                geodesic_alg_wrapper,
                tqdm(batched, total=len(batched), postfix=f"Computing Princeton benchmark for test mesh {mesh_number}")
            )
        mesh_number += 1
        break

    ##########################
    # Sorting geodesic errors
    ##########################
    geodesic_errors = np.array(geodesic_errors)
    geodesic_errors.sort()
    amt_values = geodesic_errors.shape[0]

    # Create plot-values: y-values = accuracy, x-values = geodesic errors
    arr = np.array([((i + 1) / amt_values, x) for (i, x) in zip(range(amt_values), geodesic_errors)])
    np.save(f"{file_name}.npy", arr)

    ###############################################################
    # One y-value per x-value: Take highest percentage per x-value
    ###############################################################
    unique_x_values = np.unique(arr[:, 1])
    unique_values = []
    for unique_x in unique_x_values:
        unique_values.append(arr[np.where(arr[:, 1] == unique_x)[0][-1]])
    unique_values = np.array(unique_values)

    ###########
    # Plotting
    ###########
    plt.plot(unique_values[:, 1], unique_values[:, 0], label=curve_label)
    plt.title(plot_title)
    plt.xlabel("geodesic error")
    plt.ylabel("% correct correspondences")
    plt.grid()
    plt.legend()
    plt.savefig(f"{file_name}.svg")
    if plot:
        plt.show()
    plt.close()


def geodesic_alg_wrapper(ground_truth_and_prediction, reference_mesh):
    """A wrapper function for PyGeodesicAlgorithmExact

    Required, since 'geodesic.PyGeodesicAlgorithmExact' can't be directly used as an argument for 'Pool.starmap'.

    Parameters
    ----------
    ground_truth_and_prediction: np.ndarray
        Simple array with two entries. First entry is the index of the ground truth vertex.
        The second entry is the index of the predicted vertex.
    reference_mesh: trimesh.Trimesh
        The triangle mesh on which the geodesic distances will be calculated.

    Returns
    -------
    float:
        The geodesic distance between the ground truth and predicted vertex.
    """
    geoalg = geodesic.PyGeodesicAlgorithmExact(reference_mesh.vertices, reference_mesh.faces)
    gt, pred = ground_truth_and_prediction
    return geoalg.geodesicDistance(pred, gt)[0]
