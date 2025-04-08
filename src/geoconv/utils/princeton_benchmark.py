from geoconv.utils.misc import normalize_mesh

from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

import pygeodesic.geodesic as geodesic
import trimesh
import numpy as np


def plot_geodesic_errors_scalar_field(mesh, geodesic_errors, cmap_str="Reds"):
    """This function plots the given geodesic errors for a given mesh.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The mesh on top of which the geodesic errors will be plotted.
    geodesic_errors: np.ndarray
        A 1D array containing scalar values that describe the geodesic error of the predictions for the respective
        vertices.
    cmap_str: str
        A string that describes the colormap to use when plotting the scalar field.
    """
    cmap = plt.get_cmap(cmap_str)
    point_cloud = trimesh.PointCloud(
        vertices=mesh.vertices, colors=cmap(geodesic_errors)
    )
    point_cloud.show()


def geodesic_alg_wrapper(ground_truth_and_prediction, reference_mesh):
    """A wrapper function for PyGeodesicAlgorithmExact

    Required, since 'geodesic.PyGeodesicAlgorithmExact' can't be directly used as an argument for 'Pool.starmap'.

    Parameters
    ----------
    ground_truth_and_prediction: np.ndarray
        Simple 1D array with two entries. First entry is the index of the ground truth vertex.
        The second entry is the index of the predicted vertex.
    reference_mesh: trimesh.Trimesh
        The triangle mesh on which the geodesic distances will be calculated.

    Returns
    -------
    float:
        The geodesic distance between the ground truth and predicted vertex.
    """
    geoalg = geodesic.PyGeodesicAlgorithmExact(
        reference_mesh.vertices, reference_mesh.faces
    )
    gt, pred = ground_truth_and_prediction
    return geoalg.geodesicDistance(pred, gt)[0]


def princeton_benchmark(
    imcnn,
    test_dataset,
    ref_mesh_path,
    file_name,
    normalize=True,
    plot_title="Princeton Benchmark",
    curve_label=None,
    plot=True,
    processes=1,
    geodesic_diameter=None,
    pytorch_model=False,
    add_csv=True,
):
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
    add_csv: bool
        Whether to store the plot additionally as a csv-file.
    """
    ###########################
    # Normalize reference mesh
    ###########################
    reference_mesh = trimesh.load_mesh(ref_mesh_path)
    if normalize:
        reference_mesh, _ = normalize_mesh(
            reference_mesh, geodesic_diameter=geodesic_diameter
        )

    #######################################################
    # Compute geodesic errors on normalized reference mesh
    #######################################################
    mesh_number = 0
    geodesic_errors = []
    for (signal, barycentric), ground_truth in test_dataset:
        # Get predictions of the model
        if pytorch_model:
            prediction = imcnn([signal, barycentric]).cpu()
            ground_truth = ground_truth.cpu()
        else:
            prediction = imcnn([signal, barycentric])

        prediction = np.array(prediction).argmax(axis=-1)

        # Create ground-truth/prediction-pairs and prepare data for multiprocessing
        # TODO: Account for batch sizes > 1!  'np.stack([ground_truth, prediction], axis=-1)-->[0]<--'
        batched = [
            (data, reference_mesh)
            for data in np.stack([ground_truth, prediction], axis=-1)[0]
        ]

        # Calculate geodesic distance of ground-truth to prediction on the given reference mesh
        with Pool(processes) as p:
            mesh_errors = p.starmap(
                geodesic_alg_wrapper,
                tqdm(
                    batched,
                    total=len(batched),
                    postfix=f"Computing Princeton benchmark for test mesh {mesh_number}",
                ),
            )
        geodesic_errors.append(mesh_errors)
        mesh_number += 1

    ###########################
    # Princeton benchmark plot
    ###########################
    # We are interested in the percentage 'p' of vertices that are predicted within a vicinity of geodesic error 'x'.
    # That is, when [e_1, ..., e_n] represent geodesic errors for all 'n' predictions, we have to calculate:
    # p(x) = len([e_i | e_i in [e_1, ..., e_n], e_i <= x]) / n

    # Geodesic errors: [e_1, ..., e_n]
    geodesic_errors = np.array(geodesic_errors).reshape(-1)
    np.save(f"{file_name}_geodesic_errors.npy", geodesic_errors)

    # As x-values we select the uniquely occurring geodesic errors in [e_1, ..., e_n]
    n = geodesic_errors.shape[0]
    x_values = np.unique(geodesic_errors)
    y_values = []
    for x in x_values:
        y_values.append(len([e for e in geodesic_errors if e <= x]) / n)
    y_values = np.array(y_values)

    x_y_values = np.stack([x_values, y_values], axis=-1)
    np.save(f"{file_name}_plot_values.npy", x_y_values)
    if add_csv:
        np.savetxt(
            f"{file_name}_plot_values.csv", x_y_values, delimiter=",", fmt="%.18f"
        )

    ###########
    # Plotting
    ###########
    if curve_label is None:
        plt.step(x_values, y_values, where="post")
    else:
        plt.step(x_values, y_values, where="post", label=curve_label)
        plt.legend()
    plt.title(plot_title)
    plt.xlabel("geodesic error")
    plt.ylabel("% correct correspondences")
    plt.grid()
    plt.savefig(f"{file_name}.svg")
    if plot:
        plt.show()
    plt.close()


def compute_auc(pb_plot):
    """Computes the area under the princeton benchmark plot.

    The higher the number, the more precise the benchmarked model.

    Parameters
    ----------
    pb_plot: np.ndarray
        A 2D array that contains the values used for the princeton benchmark plot.

    Returns
    -------
    float:
        A real number describing how precise the benchmarked model is. If the mesh on which the princeton benchmark was
        computed has a normalized geodesic diameter, the return value will be in between the interval [0, 1].
    """
    rectangle_areas = []
    for i in range(pb_plot.shape[0] - 1):
        del_x = pb_plot[i + 1, 0] - pb_plot[i, 0]
        y = pb_plot[i, 1]
        rectangle_areas.append(del_x * y)
    return np.sum(rectangle_areas)
