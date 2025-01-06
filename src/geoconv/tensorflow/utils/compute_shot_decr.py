import tensorflow as tf
import numpy as np


def determine_central_values(start, stop, divisions):
    # a range of n + 1 values has n bins
    central_values = tf.linspace(start=start, stop=stop, num=divisions + 1)[:-1]
    histogram_step_size = tf.math.abs(central_values[0] - central_values[1])
    central_values = central_values + histogram_step_size / 2
    return central_values, histogram_step_size


@tf.function(jit_compile=True)
def shot_descr(neighborhoods,
               normals,
               neighborhood_indices,
               radius,
               azimuth_divisions=8,
               elevation_divisions=2,
               radial_divisions=2,
               histogram_bins=11):
    # Omit origin
    neighborhoods = neighborhoods[:, 1:, :]
    neighborhood_indices = neighborhood_indices[:, 1:]

    # Compute spherical coordinates of vertices in neighborhoods
    v_radial = tf.linalg.norm(neighborhoods, axis=-1)
    v_elevation = tf.math.acos(neighborhoods[:, :, 2] / v_radial)
    v_azimuth = tf.math.atan2(neighborhoods[:, :, 1], neighborhoods[:, :, 0]) + np.pi

    # Bin spherical coordinates of vertices into spherical grid
    radial_histogram = tf.histogram_fixed_width_bins(v_radial, [0., radius], nbins=radial_divisions)
    elevation_histogram = tf.histogram_fixed_width_bins(v_elevation, [0., np.pi], nbins=elevation_divisions)
    azimuth_histogram = tf.histogram_fixed_width_bins(v_azimuth, [0., 2 * np.pi], nbins=azimuth_divisions)
    binned_vertices = tf.stack([azimuth_histogram, elevation_histogram, radial_histogram], axis=-1)

    # Compute inner product of vertex-normals from vertices in same bins with z-axis of lrf
    neighborhood_normals = tf.gather(normals, neighborhood_indices)
    cosines = tf.einsum("vi,vni->vn", normals, neighborhood_normals)
    cosine_bins = tf.histogram_fixed_width_bins(cosines, [-1., 1.], nbins=histogram_bins)

    # cosine_bins: '(vertex, neighbor, vertex-index and sphere-bin-index (3D) and histogram-index)'
    neighborhood_shape = tf.shape(neighborhood_indices)
    cosine_bins = tf.concat([binned_vertices, tf.expand_dims(cosine_bins, axis=-1)], axis=-1)
    cosine_bins = tf.concat(
        [
            tf.tile(
                tf.reshape(tf.range(neighborhood_shape[0]), (neighborhood_shape[0], 1, 1)),
                multiples=(1, neighborhood_shape[1], 1)
            ),
            cosine_bins
        ], axis=-1
    )

    # Create histogram tensor and fill it by incrementing indexed bins
    histogram = tf.zeros(
        shape=(tf.shape(neighborhoods)[0], azimuth_divisions, elevation_divisions, radial_divisions, histogram_bins)
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
    central_values, step_size = determine_central_values(start=-1., stop=1., divisions=histogram_bins)
    d = tf.abs(cosines - tf.gather(central_values, cosine_bins[:, :, histogram_colon])) / step_size

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(-tf.square(tf.expand_dims(cosines, axis=-1) - central_values), k=2)[1][:, :, 1]
    d = tf.abs(cosines - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat([cosine_bins[:, :, :histogram_colon], tf.expand_dims(closest_neighbor, axis=-1)], axis=-1),
        1 - d
    )

    ################################
    # Azimuth volumes interpolation
    ################################
    central_values, step_size = determine_central_values(
        start=0., stop=2 * np.pi, divisions=azimuth_divisions
    )
    d = tf.abs(v_azimuth - tf.gather(central_values, cosine_bins[:, :, azimuth_colon])) / step_size

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(-tf.square(tf.expand_dims(v_azimuth, axis=-1) - central_values), k=2)[1][:, :, 1]
    d = tf.abs(v_azimuth - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :azimuth_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
                cosine_bins[:, :, azimuth_colon + 1:]
            ], axis=-1
        ),
        1 - d
    )

    ##################################
    # Elevation volumes interpolation
    ##################################
    central_values, step_size = determine_central_values(
        start=0., stop=np.pi, divisions=elevation_divisions
    )
    d = tf.abs(v_elevation - tf.gather(central_values, cosine_bins[:, :, elevation_colon])) / step_size

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(-tf.square(tf.expand_dims(v_elevation, axis=-1) - central_values), k=2)[1][:, :, 1]
    d = tf.abs(v_elevation - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :elevation_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
                cosine_bins[:, :, elevation_colon + 1:]
            ], axis=-1
        ),
        1 - d
    )

    ###############################
    # Radial volumes interpolation
    ###############################
    central_values, step_size = determine_central_values(
        start=0., stop=radius, divisions=radial_divisions
    )
    d = tf.abs(v_radial - tf.gather(central_values, cosine_bins[:, :, radial_colon])) / step_size

    # Increment histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = tf.math.top_k(-tf.square(tf.expand_dims(v_radial, axis=-1) - central_values), k=2)[1][:, :, 1]
    d = tf.abs(v_radial - tf.gather(central_values, closest_neighbor)) / step_size

    # Increment neighboring histogram bins by 1 - d
    histogram = tf.tensor_scatter_nd_add(
        histogram,
        tf.concat(
            [
                cosine_bins[:, :, :radial_colon],
                tf.expand_dims(closest_neighbor, axis=-1),
                cosine_bins[:, :, radial_colon + 1:]
            ], axis=-1
        ),
        1 - d
    )

    #######################
    # Form SHOT-descriptor
    #######################
    # Reshape histogram into vector
    histogram = tf.reshape(histogram, (neighborhood_shape[0], -1))

    # Normalize descriptor to have length 1
    histogram = histogram * (1 / tf.expand_dims(tf.linalg.norm(histogram, axis=-1), axis=-1))

    return histogram
