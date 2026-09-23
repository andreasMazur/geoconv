from geoconv.pytorch.utils.compute_shot_lrf import tensor_scatter_nd_add_

import torch
import numpy as np


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
    (torch.Tensor, float):
        The central values of the histogram bins and the step size between any two central values.
    """
    # A range of n + 1 values has n bins
    central_values = torch.linspace(start=start, end=stop, steps=n_bins + 1)[:-1]
    histogram_step_size = torch.abs(central_values[0] - central_values[1])
    central_values = central_values + histogram_step_size / 2
    return central_values, histogram_step_size


def shot_descr(neighborhoods,
               normals,
               neighborhood_indices,
               radius,
               azimuth_bins=8,
               elevation_bins=2,
               radial_bins=2,
               histogram_bins=11):
    """This function computes SHOT-descriptor.

    SHOT-descriptor have been introduced in:
    > [SHOT: Unique signatures of histograms for surface and texture
     description.](https://doi.org/10.1016/j.cviu.2014.04.011)
    > Salti, Samuele, Federico Tombari, and Luigi Di Stefano.

    Parameters
    ----------
    neighborhoods: torch.Tensor
        A rank-3 tensor of shape (n_vertices, n_neighbors, 3) containing the cartesian coordinates of neighbors.
    normals: torch.Tensor
        A rank-2 tensor of shape (n_vertices, 3) containing the normals of the vertices.
    neighborhood_indices: torch.Tensor
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
    torch.Tensor:
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
    # 'v_radial': (n_vertices, n_neighbors)
    v_radial = torch.linalg.norm(neighborhoods, axis=-1)

    # 'v_elevation': (n_vertices, n_neighbors)
    v_elevation = torch.acos(  # machine accuracy sometimes return slightly larger/smaller values than allowed
        torch.clamp(neighborhoods[:, :, 2] / v_radial, min=-1.0, max=1.0)
    )

    # 'v_azimuth': (n_vertices, n_neighbors)
    v_azimuth = torch.atan2(neighborhoods[:, :, 1], neighborhoods[:, :, 0]) + np.pi

    # Bin spherical coordinates
    # 'radial_boundaries': (n_radial_bins - 1,)
    radial_boundaries = torch.linspace(0.0, radius, radial_bins + 1, device=neighborhoods.device)[1:-1]

    # 'elevation_boundaries': (n_elevation_bins - 1,)
    elevation_boundaries = torch.linspace(0.0, np.pi, elevation_bins + 1, device=neighborhoods.device)[1:-1]

    # 'azimuth_boundaries': (n_azimuth_bins - 1,)
    azimuth_boundaries = torch.linspace(0.0, 2 * np.pi, azimuth_bins + 1, device=neighborhoods.device)[1:-1]

    # Bin spherical coordinates of vertices into spherical grid
    # '?_histogram': (n_vertices, n_neighbors)
    radial_histogram = torch.bucketize(input=v_radial, boundaries=radial_boundaries)
    elevation_histogram = torch.bucketize(input=v_elevation, boundaries=elevation_boundaries)
    azimuth_histogram = torch.bucketize(input=v_azimuth, boundaries=azimuth_boundaries)

    # 'sphere_bins': (n_vertices, n_neighbors, 3)
    sphere_bins = torch.stack([azimuth_histogram, elevation_histogram, radial_histogram], dim=-1)

    # Compute inner product of vertex-normals from vertices in same bins with z-axis of lrf
    # 'neighborhood_normals': (n_vertices, n_neighbors, 3)
    neighborhood_normals = normals[neighborhood_indices]

    # 'cosines': (n_vertices, n_neighbors)
    cosines = torch.einsum("vi,vni->vn", normals, neighborhood_normals)

    # 'histogram_boundaries': (n_histogram_bins - 1,)
    histogram_boundaries = torch.linspace(-1.0, 1.0, histogram_bins + 1, device=neighborhood_indices.device)[1:-1]

    # 'cosine_bins': (n_vertices, n_neighbors)
    cosine_bins = torch.bucketize(cosines, boundaries=histogram_boundaries)

    # cosine_bins: '(vertex, neighbor, sphere-bin-index AND histogram-index => 3 + 1 = 4)'
    cosine_bins = torch.concat([sphere_bins, cosine_bins.unsqueeze(dim=-1)], dim=-1)

    # cosine_bins: '(vertex, neighbor, vertex-index AND sphere-bin-index AND histogram-index => 1 + 4 = 5)'
    neighborhood_shape = neighborhood_indices.size()
    cosine_bins = torch.concat(
        [
            torch.arange(neighborhood_shape[0])[:, None, None].repeat(1, neighborhood_shape[1], 1),
            cosine_bins
        ],
        dim=-1
    )

    # Create histogram tensor and fill it by incrementing indexed bins
    histogram = torch.zeros(
        (
            neighborhoods.size(0),
            azimuth_bins,
            elevation_bins,
            radial_bins,
            histogram_bins,
        )
    )

    ##############################
    # Quadrilateral interpolation
    ##############################
    azimuth_column = 1
    elevation_column = 2
    radial_column = 3
    histogram_column = 4

    ###############################
    # Histogram bins interpolation
    ###############################
    # 'central_values': (n_histogram_bins,)
    # 'step_size'     : ()
    central_values, step_size = determine_central_values(start=-1.0, stop=1.0, n_bins=histogram_bins)

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(cosines - central_values[cosine_bins[:, :, histogram_column]]) / step_size

    # Increment histogram bins by 1 - d
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    # 'closest_neighbor': (n_vertices, n_neighbors)
    closest_neighbor = torch.topk(-torch.square(cosines[..., None] - central_values), k=2)[1][..., 1]

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(cosines - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,
        torch.concat(
            [
                cosine_bins[:, :, :histogram_column],
                closest_neighbor[..., None]
            ],
            dim=-1
        ),
        1 - d
    )

    ################################
    # Azimuth volumes interpolation
    ################################
    # 'central_values': (n_azimuth_bins,)
    # 'step_size'     : ()
    central_values, step_size = determine_central_values(start=0.0, stop=2 * np.pi, n_bins=azimuth_bins)

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(v_azimuth - central_values[cosine_bins[:, :, azimuth_column]]) / step_size

    # Increment histogram bins by 1 - d
    # 'histogram':   (n_vertices, n_azimuth_bins, n_elevation_bins, n_radial_bins, n_histogram_bins)
    # 'cosine_bins': (n_vertices, n_azimuth_bins)
    # 'd':           (n_vertices, n_neighbors)
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    # 'closest_neighbor': (n_vertices, n_neighbors)
    closest_neighbor = torch.topk(-torch.square(v_azimuth.unsqueeze(dim=-1) - central_values), k=2)[1][:, :, 1]

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(v_azimuth - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,  # (n_vertices, n_azimuth_bins, n_elevation_bins, n_radial_bins, n_histogram_bins)
        torch.concat(  # (n_vertices, n_neighbors, 5)
            [
                cosine_bins[:, :, :azimuth_column],
                closest_neighbor.unsqueeze(dim=-1),
                cosine_bins[:, :, azimuth_column + 1:]
            ],
            dim=-1
        ),
        1 - d  # (n_vertices, n_neighbors)
    )

    ##################################
    # Elevation volumes interpolation
    ##################################
    # 'central_values': (n_elevation_bins,)
    # 'step_size'     : ()
    central_values, step_size = determine_central_values(start=0.0, stop=np.pi, n_bins=elevation_bins)

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(v_elevation - central_values[cosine_bins[:, :, elevation_column]]) / step_size

    # Increment histogram bins by 1 - d
    # 'histogram':   (n_vertices, n_azimuth_bins, n_elevation_bins, n_radial_bins, n_histogram_bins)
    # 'cosine_bins': (n_vertices, n_elevation_bins)
    # 'd':           (n_vertices, n_neighbors)
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    # 'closest_neighbor': (n_vertices, n_neighbors)
    closest_neighbor = torch.topk(-torch.square(v_elevation.unsqueeze(dim=-1) - central_values), k=2)[1][:, :, 1]

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(v_elevation - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,  # (n_vertices, n_azimuth_bins, n_elevation_bins, n_radial_bins, n_histogram_bins)
        torch.concat(  # (n_vertices, n_neighbors, 5)
            [
                cosine_bins[:, :, :elevation_column],
                closest_neighbor.unsqueeze(dim=-1),
                cosine_bins[:, :, elevation_column + 1:]
            ],
            dim=-1
        ),
        1 - d  # (n_vertices, n_neighbors)
    )

    ###############################
    # Radial volumes interpolation
    ###############################
    # 'central_values': (n_radial_bins,)
    # 'step_size'     : ()
    central_values, step_size = determine_central_values(start=0.0, stop=radius, n_bins=radial_bins)

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(v_radial - central_values[cosine_bins[:, :, radial_column]]) / step_size

    # Increment histogram bins by 1 - d
    # 'histogram':   (n_vertices, n_azimuth_bins, n_elevation_bins, n_radial_bins, n_histogram_bins)
    # 'cosine_bins': (n_vertices, n_radial_bins)
    # 'd':           (n_vertices, n_neighbors)
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    # 'closest_neighbor': (n_vertices, n_neighbors)
    closest_neighbor = torch.topk(-torch.square(v_radial.unsqueeze(dim=-1) - central_values), k=2)[1][:, :, 1]

    # 'd': (n_vertices, n_neighbors)
    d = torch.abs(v_radial - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,  # (n_vertices, n_azimuth_bins, n_elevation_bins, n_radial_bins, n_histogram_bins)
        torch.concat(  # (n_vertices, n_neighbors, 5)
            [
                cosine_bins[:, :, :radial_column],
                closest_neighbor.unsqueeze(dim=-1),
                cosine_bins[:, :, radial_column + 1:]
            ],
            dim=-1
        ),
        1 - d  # (n_vertices, n_neighbors)
    )

    #########################################
    # Reshape histogram into SHOT-descriptor
    #########################################
    # Reshape histogram into vector
    histogram = torch.reshape(histogram, (neighborhood_shape[0], -1))

    # Normalize descriptor to have length 1
    return histogram / torch.linalg.norm(histogram, axis=-1, keepdims=True)
