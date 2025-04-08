import tensorflow as tf
import numpy as np


@tf.function(jit_compile=True)
def determine_central_values(start, stop, n_bins):
    """Determines the central values within the bins.

    Parameters
    ----------
    start: float
        The start x-value of the histogram.
    stop: float
        The stop x-value of the histogram.
    n_bins: int
        The amount of bins.

    Returns
    -------
    (tf.Tensor, float):
        The central values of the histogram bins and the step size between any two central values.
    """
    # A range of n + 1 values has n bins
    central_values = tf.linspace(start=start, stop=stop, num=n_bins + 1)[:-1]
    histogram_step_size = tf.math.abs(central_values[0] - central_values[1])
    central_values = central_values + histogram_step_size / 2
    return central_values, histogram_step_size


@tf.function(jit_compile=True)
def shot_descr(
    neighborhoods,
    normals,
    neighborhood_indices,
    radius,
    azimuth_bins=8,
    elevation_bins=2,
    radial_bins=2,
    histogram_bins=11,
):
    """This function computes SHOT-descriptor.

    SHOT-descriptor have been introduced in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhoods: tf.Tensor
        A rank-3 tensor of shape (n_vertices, n_neighbors, 3) containing the cartesian coordinates of neighbors.
    normals: tf.Tensor
        A rank-2 tensor of shape (n_vertices, 3) containing the normals of the vertices.
    neighborhood_indices: tf.Tensor
        A rank-2 tensor of shape (n_vertices, n_neighbors) containing the indices of the neighbors.
    radius: float
        The radius for the sphere used to compute the SHOT-descriptor.
    azimuth_bins: int
        The amount of bins along the azimuth direction.
    elevation_bins: int
        The amount of bins along the elevation direction.
    radial_bins: int
        The amount of bins along the radial direction.
    histogram_bins:
        The amount of bins in the histogram.

    Returns
    -------
    tf.Tensor:
        A rank-2 tensor of shape
            (n_vertices, azimuth_bins * elevation_bins * radial_bins * histogram_bins)
        containing the SHOT-descriptor for each vertex.
    """
    ########################################################################
    # Determine into which spherical- and histogram bins the neighbors fall
    ########################################################################
    # Omit origin
    neighborhoods = neighborhoods[:, 1:, :]
    neighborhood_indices = neighborhood_indices[:, 1:]

    # Compute spherical coordinates of vertices in neighborhoods
    v_radial = tf.linalg.norm(neighborhoods, axis=-1)
    v_elevation = tf.math.acos(  # machine accuracy sometimes return slightly larger/smaller values than allowed
        tf.clip_by_value(
            neighborhoods[:, :, 2] / v_radial, clip_value_min=-1.0, clip_value_max=1.0
        )
    )
    v_azimuth = tf.math.atan2(neighborhoods[:, :, 1], neighborhoods[:, :, 0]) + np.pi

    # Bin spherical coordinates of vertices into spherical grid
    # 'x_histogram': (n_vertices, n_neighbors)
    radial_histogram = tf.histogram_fixed_width_bins(
        v_radial, [0.0, radius], nbins=radial_bins
    )
    elevation_histogram = tf.histogram_fixed_width_bins(
        v_elevation, [0.0, np.pi], nbins=elevation_bins
    )
    azimuth_histogram = tf.histogram_fixed_width_bins(
        v_azimuth, [0.0, 2 * np.pi], nbins=azimuth_bins
    )
    # 'sphere_bins': (n_vertices, n_neighbors, 3)
    sphere_bins = tf.stack(
        [azimuth_histogram, elevation_histogram, radial_histogram], axis=-1
    )

    # Compute inner product of vertex-normals from vertices in same bins with z-axis of lrf
    # 'neighborhood_normals': (n_vertices, n_neighbors, 3)
    neighborhood_normals = tf.gather(normals, neighborhood_indices)
    # 'cosines': (n_vertices, n_neighbors)
    cosines = tf.einsum("vi,vni->vn", normals, neighborhood_normals)
    # 'cosine_bins': (n_vertices, n_neighbors)
    cosine_bins = tf.histogram_fixed_width_bins(
        cosines, [-1.0, 1.0], nbins=histogram_bins
    )

    # cosine_bins: '(vertex, neighbor, sphere-bin-index AND histogram-index => 3 + 1 = 4)'
    neighborhood_shape = tf.shape(neighborhood_indices)
    cosine_bins = tf.concat(
        [sphere_bins, tf.expand_dims(cosine_bins, axis=-1)], axis=-1
    )

    # cosine_bins: '(vertex, neighbor, vertex-index AND sphere-bin-index AND histogram-index => 1 + 4 = 5)'
    cosine_bins = tf.concat(
        [
            tf.tile(
                tf.reshape(
                    tf.range(neighborhood_shape[0]), (neighborhood_shape[0], 1, 1)
                ),
                multiples=(1, neighborhood_shape[1], 1),
            ),
            cosine_bins,
        ],
        axis=-1,
    )

    # Create histogram tensor and fill it by incrementing indexed bins
    histogram = tf.zeros(
        shape=(
            tf.shape(neighborhoods)[0],
            azimuth_bins,
            elevation_bins,
            radial_bins,
            histogram_bins,
        )
    )

    ##############################
    # Quadrilateral interpolation
    ##############################
    azimuth_colon = 1
    elevation_colon = 2
    radial_colon = 3
    histogram_colon = 4

    ###############################
    # Histogram bins interpolation
    ###############################
    central_values, step_size = determine_central_values(
        start=-1.0, stop=1.0, n_bins=histogram_bins
    )
    d = (
        tf.abs(cosines - tf.gather(central_values, cosine_bins[:, :, histogram_colon])) / step_size
    )

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(
        -tf.square(tf.expand_dims(cosines, axis=-1) - central_values), k=2
    )[1][:, :, 1]
    d = tf.abs(cosines - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :histogram_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
            ],
            axis=-1,
        ),
        1 - d,
    )

    ################################
    # Azimuth volumes interpolation
    ################################
    central_values, step_size = determine_central_values(
        start=0.0, stop=2 * np.pi, n_bins=azimuth_bins
    )
    d = (
        tf.abs(v_azimuth - tf.gather(central_values, cosine_bins[:, :, azimuth_colon])) / step_size
    )

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(
        -tf.square(tf.expand_dims(v_azimuth, axis=-1) - central_values), k=2
    )[1][:, :, 1]
    d = tf.abs(v_azimuth - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :azimuth_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
                cosine_bins[:, :, azimuth_colon + 1 :],
            ],
            axis=-1,
        ),
        1 - d,
    )

    ##################################
    # Elevation volumes interpolation
    ##################################
    central_values, step_size = determine_central_values(
        start=0.0, stop=np.pi, n_bins=elevation_bins
    )
    d = (
        tf.abs(
            v_elevation - tf.gather(central_values, cosine_bins[:, :, elevation_colon])
        ) / step_size
    )

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(
        -tf.square(tf.expand_dims(v_elevation, axis=-1) - central_values), k=2
    )[1][:, :, 1]
    d = tf.abs(v_elevation - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :elevation_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
                cosine_bins[:, :, elevation_colon + 1 :],
            ],
            axis=-1,
        ),
        1 - d,
    )

    ###############################
    # Radial volumes interpolation
    ###############################
    central_values, step_size = determine_central_values(
        start=0.0, stop=radius, n_bins=radial_bins
    )
    d = (
        tf.abs(v_radial - tf.gather(central_values, cosine_bins[:, :, radial_colon])) / step_size
    )

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(
        -tf.square(tf.expand_dims(v_radial, axis=-1) - central_values), k=2
    )[1][:, :, 1]
    d = tf.abs(v_radial - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :radial_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
                cosine_bins[:, :, radial_colon + 1 :],
            ],
            axis=-1,
        ),
        1 - d,
    )

    #########################################
    # Reshape histogram into SHOT-descriptor
    #########################################
    # Reshape histogram into vector
    histogram = tf.reshape(histogram, (neighborhood_shape[0], -1))

    # Normalize descriptor to have length 1
    return histogram / tf.linalg.norm(histogram, axis=-1, keepdims=True)
