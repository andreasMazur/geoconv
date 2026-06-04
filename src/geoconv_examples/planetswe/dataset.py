from geoconv.preprocessing.distance_computation import normalize_shape

from tqdm import tqdm

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


def create_planetswe_sphere(path, normalization_method="hdm", processes=1):
    """Creates a sphere from the longitude and colatitude angles contained in the planetswe dataset.

    Parameters
    ----------
    path: str
        The path to the directory "planetswe".
    normalization_method: str
        The method used to normalize the shape. Either "fmm" or "hdm".
    processes: int
        The number of concurrent processes used to determine the geodesic diameter for shape normalization.

    Returns
    -------
    trimesh.Trimesh:
        The sphere triangle mesh for the planetswe dataset.
    """
    file_path = f"{path}/data/train/planetswe_IC00_s1.hdf5"
    file_content = h5py.File(file_path, "r")
    longitude_phi = np.array(file_content["dimensions"]["phi"])
    colatitude_theta = np.array(file_content["dimensions"]["theta"])
    sphere = create_sphere(colatitude_theta, longitude_phi)
    sphere, geodesic_diameter = normalize_shape(
        sphere,
        method=normalization_method,
        processes=processes
    )
    return sphere


def planetswe_hdf5_reader(file_content, normalize=False):
    """Reads the groups of interest for one HDF5 file in the planetswe dataset.

    Parameters
    ----------
    file_content: h5py.File
        The loaded HDF5 file.
    normalize: bool
        Whether to normalize the feature fields to their z-scores.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    """
    ### Other stored stuff ###
    # bc_phi_periodic_mask = file_content["boundary_conditions"]["phi_periodic"]["mask"]
    # bc_theta_open_mask = file_content["boundary_conditions"]["theta_open"]["mask"]
    # dimensions_time = np.array(file_content["dimensions"]["time"])  # shape: [1008]

    ### Stuff of interest ###
    # shape: [512] (longitude) range(0.0, 6.2709136) [0, 2pi[
    longitude_phi = np.array(file_content["dimensions"]["phi"])
    # shape: [256] (colatitude) range(0.009375532, 3.1322172) [0, pi[
    colatitude_theta = np.array(file_content["dimensions"]["theta"])

    # Load heights and velocities
    field_height = file_content["t0_fields"]["height"][0]  # shape: [1008, 256, 512]
    field_velocity = file_content["t1_fields"]["velocity"][0]  # shape: [1008, 256, 512, 2]

    ### Normalize (z-score) ###
    if normalize:
        height_mean = field_height.mean()
        height_std = field_height.std()
        velocity_mean = field_velocity.mean()
        velocity_std = field_velocity.std()

        field_height = (field_height - height_mean) / height_std
        field_velocity = (field_velocity - velocity_mean) / velocity_std

    return longitude_phi, colatitude_theta, field_height, field_velocity


def planetswe_raw_data_generator(path,
                                 split,
                                 normalize_features=True,
                                 return_sphere=False,
                                 add_input_zero_dim=False,
                                 return_filename=False):
    """Reads the hdf5-files from the planetswe dataset.

    Parameters
    ----------
    path: str
        The path to the directory "planetswe".
    split: str
        Either "train", "valid" or "test".
    normalize_features: bool
        Whether to normalize the height and velocity fields.
    return_sphere: bool
        Whether to return the point-cloud sphere together with other features.
    add_input_zero_dim: bool
        If 'True', adds a zero to each feature vector to return an even amount of features. This is required for
        architectures that expect complex input features.
    return_filename: bool
        Whether to return the filename of the feature fields.

    Returns
    -------
    np.ndarray | (np.ndarray, np.ndarray):
        The feature fields and the point-cloud for the sphere if wanted.
    """
    split_dir = f"{path}/data/{split}"
    split_content = [f"{path}/data/{split}/{f}" for f in os.listdir(split_dir)]
    split_content.sort(key=lambda x: "_".join(os.path.basename(x).replace(".", "_").split("_")[1:-1]))
    spherical_point_cloud = None

    for file_idx, file_path in enumerate(split_content):
        ### Read complete file ###
        file_content = h5py.File(file_path, "r")
        longitude_phi, colatitude_theta, field_height, field_velocity = planetswe_hdf5_reader(
            file_content, normalize=normalize_features
        )

        ### Create 3D spherical coordinates in case non has been loaded so far ###
        if return_sphere and spherical_point_cloud is None:
            sphere = create_sphere(colatitude_theta, longitude_phi)
            spherical_point_cloud = np.array(sphere.vertices)

        ### Concatenate velocities and heights to single feature vectors ###
        feature_vectors = np.concatenate(
            [field_velocity.reshape(1008, -1, 2), field_height.reshape(1008, -1, 1)], axis=-1
        )

        ### Add zero dimension in case even amount of features are wished for ###
        if add_input_zero_dim:
            feature_vectors = np.concatenate(
                [feature_vectors, np.zeros(feature_vectors.shape[:-1] + (1,))], axis=-1
            )

        ### Gather yield values ###
        yield_values = (feature_vectors,)

        if return_sphere:
            yield_values = (spherical_point_cloud,) + yield_values

        if return_filename:
            yield_values = yield_values + (os.path.basename(file_path),)
        yield yield_values


def generator(bc_path, swe_path, set_type, return_rotations=False, add_input_zero_dim=False):
    """Returns a 'generator'-object for the planetswe dataset.

    Parameters
    ----------
    bc_path: str
        The path to the zip file that contains all barycentric coordinates chunks.
    swe_path: str
        The path to the downloaded planetswe dataset.
    set_type: str
        The set type. Either: 'train', 'valid' or 'test'.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.
    add_input_zero_dim: bool
        If 'True', adds a zero to each feature vector to return an even amount of features. This is required for
        architectures that expect complex input features.

    Returns
    -------
    generator:
        A planetswe generator.
    """
    if isinstance(bc_path, bytes):
        bc_path = bc_path.decode("utf-8")
    if isinstance(swe_path, bytes):
        swe_path = swe_path.decode("utf-8")
    if isinstance(set_type, bytes):
        set_type = set_type.decode("utf-8")

    # 1.) Load barycentric coordinates for the sphere
    barycentric_zip = np.load(bc_path)
    barycentric_coordinate_chunks = [f for f in barycentric_zip.files if "barycentric_coordinates" in f]
    barycentric_coordinate_chunks.sort(key=lambda x: int(x.split("/")[0].split("_")[-2]))

    cut_idx = 3 if return_rotations else 2
    barycentric_coordinates = np.concatenate(
        [barycentric_zip[f] for f in tqdm(barycentric_coordinate_chunks, postfix="Loading barycentric coordinates..")],
        axis=0
    )[..., :cut_idx]

    # 2.) Load the feature fields from planetswe
    swe_raw_generator = planetswe_raw_data_generator(
        path=swe_path,
        split=set_type,
        normalize_features=True,
        return_sphere=False,
        add_input_zero_dim=add_input_zero_dim,
        return_filename=True
    )

    # 3.) Return feature field and barycentric coordinates for one time step pair (t, t+1) at a time
    last_feature_field = None
    for (year_of_feature_fields, filename) in swe_raw_generator:
        # Remember which split pair we have
        split_number = filename.split(".")[0].split("_")[-1]

        # First splits have no predecessor feature fields
        last_feature_field = None if split_number == "s1" else last_feature_field

        # Handle cross-split time-steps
        if last_feature_field is not None:
            yield (last_feature_field, barycentric_coordinates), year_of_feature_fields[0, :, :3]

        for time_idx in range(year_of_feature_fields.shape[0]):
            # Last element in time trajectory [s1, s2, s3]
            if time_idx + 1 == year_of_feature_fields.shape[0] and split_number == "s3":
                break
            # Last time step within a split sX of trajectory [s1, s2, s3]
            elif time_idx + 1 == year_of_feature_fields.shape[0]:
                last_feature_field = year_of_feature_fields[time_idx]
            # Regular time step within a split sX of trajectory [s1, s2, s3]
            else:
                t_feature_field = year_of_feature_fields[time_idx]
                t_next_feature_field = year_of_feature_fields[time_idx + 1, :, :3]
                yield (t_feature_field, barycentric_coordinates), t_next_feature_field


def dataset(bc_path, swe_path, set_type, batch_size=1, return_rotations=False, add_input_zero_dim=False):
    """Returns a 'tensorflow dataset'-object for the planetswe dataset.

    Parameters
    ----------
    bc_path: str
        The path to the zip file that contains all barycentric coordinates chunks.
    swe_path: str
        The path to the downloaded planetswe dataset.
    set_type: str
        The set type. Either: 'train', 'valid' or 'test'.
    return_rotations: bool
        Whether to return the rotation angles for the parallel transport.
    batch_size: int
        The batch size.
    add_input_zero_dim: bool
        If 'True', adds a zero to each feature vector to return an even amount of features. This is required for
        architectures that expect complex input features.

    Returns
    -------
    generator:
        A planetswe dataset.
    """
    if return_rotations:
        bc_shape = (3, 3)
    else:
        bc_shape = (3, 2)
    input_feature_dim = 4 if add_input_zero_dim else 3
    n_radial, n_angular = os.path.basename(bc_path).split(".")[0].split("_")[-2:]

    output_signature = (
        (
            tf.TensorSpec(shape=(131072, input_feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(131072,) + (int(n_radial), int(n_angular)) + bc_shape, dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(131072, 3), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(
        generator,
        args=(bc_path, swe_path, set_type, return_rotations, add_input_zero_dim),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE).batch(batch_size)
