import tensorflow as tf
import h5py
import os
import numpy as np
import trimesh


def create_sphere(colatitude_theta, longitude_phi):
    """Creates Cartesian coordinates for colatitude and longitude spherical coordiantes.

    Parameters
    ----------
    colatitude_theta: np.ndarray
        The colatitude angles.
    longitude_phi: np.ndarray
        The longitude angles.

    Returns
    -------
    trimesh.Trimesh:
        A triangle mesh for the given spherical coordinates.
    """
    theta_grid, phi_grid = np.meshgrid(colatitude_theta, longitude_phi, indexing="ij")
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    spherical_point_cloud = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    n_colatitude_indices, n_longitude_indices = 256, 512
    faces = []
    for i in range(1, n_colatitude_indices - 2):
        for j in range(n_longitude_indices):
            p0 = i * n_longitude_indices + j
            p1 = i * n_longitude_indices + (j + 1) % n_longitude_indices
            p2 = (i + 1) * n_longitude_indices + j
            p3 = (i + 1) * n_longitude_indices + (j + 1) % n_longitude_indices
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])

    # connecting interior ring (south pole)
    for j in range(n_longitude_indices):
        p_south = j
        p_next = 1 * n_longitude_indices + j
        p_next2 = 1 * n_longitude_indices + (j + 1) % n_longitude_indices
        faces.append([p_south, p_next, p_next2])

    # connecting interior ring (north pole)
    offset = (n_colatitude_indices - 1) * n_longitude_indices
    for j in range(n_longitude_indices):
        p_north = offset + j
        p_prev = (n_colatitude_indices - 2) * n_longitude_indices + j
        p_prev2 = (n_colatitude_indices - 2) * n_longitude_indices + (j + 1) % n_longitude_indices
        faces.append([p_north, p_prev2, p_prev])

    print(f"Created spherical point cloud with {spherical_point_cloud.shape[0]} vertices and {len(faces)} faces")
    return trimesh.Trimesh(vertices=spherical_point_cloud, faces=np.array(faces), process=False)


def create_planetswe_sphere(path):
    """Creates a sphere from the longitude and colatitude angles contained in the planetswe dataset.

    Parameters
    ----------
    path: str
        The path to the directory "planetswe".

    Returns
    -------
    trimesh.Trimesh:
        The sphere triangle mesh for the planetswe dataset.
    """
    file_path = f"{path}/data/train/planetswe_IC00_s1.hdf5"
    file_content = h5py.File(file_path, "r")
    longitude_phi = np.array(file_content["dimensions"]["phi"])
    colatitude_theta = np.array(file_content["dimensions"]["theta"])
    return create_sphere(colatitude_theta, longitude_phi)


def planetswe_raw_data_generator(path, split, normalize_features=True):
    """Reads the hdf5-files from the planetswe dataset.

    Parameters
    ----------
    path: str
        The path to the directory "planetswe".
    split: str
        Either "train", "valid" or "test".
    normalize_features: bool
        Whether to normalize the height and velocity fields.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray):
        The point-cloud for the sphere, and the feature fields.
    """
    split_dir = f"{path}/data/{split}"
    split_content = [f"{path}/data/{split}/{f}" for f in os.listdir(split_dir)]
    split_content.sort(key=lambda x: "_".join(x.replace(".", "_").split("_")[1:-1]))
    spherical_point_cloud = None

    for file_path in split_content:
        # Read complete file
        file_content = h5py.File(file_path, "r")

        ### Other stored stuff ###
        # bc_phi_periodic_mask = file_content["boundary_conditions"]["phi_periodic"]["mask"]
        # bc_theta_open_mask = file_content["boundary_conditions"]["theta_open"]["mask"]
        # dimensions_time = np.array(file_content["dimensions"]["time"])  # shape: [1008]

        ### Stuff of interest ###
        # shape: [512] (longitude) range(0.0, 6.2709136) [0, 2pi[
        longitude_phi = np.array(file_content["dimensions"]["phi"])
        # shape: [256] (colatitude) range(0.009375532, 3.1322172) [0, pi[
        colatitude_theta = np.array(file_content["dimensions"]["theta"])

        # Create 3D spherical coordinates:
        if spherical_point_cloud is None:
            sphere = create_sphere(colatitude_theta, longitude_phi)
            spherical_point_cloud = np.array(sphere.vertices)

        # Load heights and velocities
        t0_field_height = file_content["t0_fields"]["height"][0]  # shape: [1008, 256, 512]
        t1_field_velocity = file_content["t1_fields"]["velocity"][0]  # shape: [1008, 256, 512, 2]

        # normalize (z-score)
        if normalize_features:
            # Numpy is way to slow, use TF instead
            height_mean = tf.reduce_mean(tf.constant(t0_field_height)).numpy()
            height_std = tf.math.reduce_std(tf.constant(t0_field_height)).numpy()
            velocity_mean = tf.reduce_mean(tf.constant(t1_field_velocity)).numpy()
            velocity_std = tf.math.reduce_std(tf.constant(t1_field_velocity)).numpy()

            t0_field_height = (t0_field_height - height_mean) / height_std
            t1_field_velocity = (t1_field_velocity - velocity_mean) / velocity_std

        yield spherical_point_cloud, t0_field_height.reshape(1008, -1), t1_field_velocity.reshape(1008, -1, 2)
