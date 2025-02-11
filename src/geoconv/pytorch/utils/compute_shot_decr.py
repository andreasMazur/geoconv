from geoconv.pytorch.utils.tensor_utils import tensor_scatter_nd_add_, histogram_fixed_width_bins

import torch
import numpy as np

from typing import Tuple

@torch.jit.script
def determine_central_values(start : float, stop : float, n_bins : int) -> Tuple[torch.Tensor, float]:
    """Determines the central values within the bins.

    Parameters
    ----------
    start: float
        The start x-value of the histogram.
    stop: float
        The stop x-value of the histogram.
    n_bins: int
        The number of bins.

    Returns
    -------
    Tuple[torch.Tensor, float]:
        The central values of the histogram bins and the step size between any two central values.
    """
    # A range of n + 1 values has n bins
    central_values = torch.linspace(start=start, end=stop, steps=n_bins + 1)[:-1]
    histogram_step_size = torch.abs(central_values[0] - central_values[1])
    central_values += histogram_step_size / 2
    return central_values, histogram_step_size

@torch.jit.script
def shot_descr(neighborhoods : torch.Tensor,
               normals : torch.Tensor,
               neighborhood_indices : torch.Tensor,
               radius : float,
               azimuth_bins : int = 8,
               elevation_bins : int = 2,
               radial_bins : int = 2,
               histogram_bins : int = 11):# -> torch.Tensor:
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
        The number of bins along the azimuth direction.
    elevation_bins: int
        The number of bins along the elevation direction.
    radial_bins: int
        The number of bins along the radial direction.
    histogram_bins: int
        The number of bins in the histogram.

    Returns
    -------
    torch.Tensor:
        A rank-3 tensor of shape (n_vertices, azimuth_bins * elevation_bins * radial_bins * histogram_bins) containing the SHOT-descriptor for each vertex.
    """
    ########################################################################
    # Determine into which spherical- and histogram bins the neighbors fall
    ########################################################################
    # Omit origin
    neighborhoods = neighborhoods[:, 1:, :]
    neighborhood_indices = neighborhood_indices[:, 1:]

    # Compute the spherical coordinates of vertices in neighborhoods
    v_radial = torch.norm(neighborhoods, dim=-1)
    v_elevation = torch.acos( # machine accuracy sometimes return slightly larger/smaller values than allowed
        torch.clamp(neighborhoods[:, :, 2] / v_radial, min=-1., max=1.)
    )
    v_azimuth = torch.atan2(neighborhoods[:, :, 1], neighborhoods[:, :, 0]) + torch.pi

    # Bin spherical coordinates of vertices into a spherical grid
    radial_histogram = histogram_fixed_width_bins(v_radial, torch.tensor([0., radius], device=v_radial.device), n_bins=radial_bins)
    elevation_histogram = histogram_fixed_width_bins(v_elevation, torch.tensor([0., torch.pi], device=v_elevation.device), n_bins=elevation_bins)
    azimuth_histogram = histogram_fixed_width_bins(v_azimuth, torch.tensor([0., 2 * torch.pi], device=v_azimuth.device), n_bins=azimuth_bins)
    binned_vertices = torch.stack([azimuth_histogram, elevation_histogram, radial_histogram], dim=-1)

    # Compute the inner product of vertex-normals from vertices in same bins with z-axis of lrf
    neighborhood_normals = normals[neighborhood_indices]
    cosines = torch.einsum('vi,vni->vn', normals, neighborhood_normals)
    cosine_bins = histogram_fixed_width_bins(cosines, torch.tensor([-1., 1.], dtype=cosines.dtype), n_bins=histogram_bins)

    # cosine_bins: (vertex, neighbor, vertex-index and sphere-bin-index (3D) and histogram-index)
    neighborhood_shape = neighborhood_indices.shape
    cosine_bins = torch.concatenate([binned_vertices, cosine_bins.unsqueeze(-1)], dim=-1)
    cosine_bins = torch.concatenate(
        [
            torch.arange(neighborhood_shape[0], device=cosine_bins.device).reshape(neighborhood_shape[0], 1, 1).repeat(1, neighborhood_shape[1], 1),
            cosine_bins
        ], dim=-1
    )

    # Create histogram tensor and fill it by incrementing indexed bins
    histogram = torch.zeros(
        (neighborhoods.shape[0], azimuth_bins, elevation_bins, radial_bins, histogram_bins)
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
    central_values, step_size = determine_central_values(start=-1., stop=1., n_bins=histogram_bins)
    d = torch.abs(cosines - central_values[cosine_bins[:, :, histogram_colon]]) / step_size

    # Increment histogram bins by 1 - d
    print(f"[pt] histogram shape: {histogram.shape}; cosine_bins shape: {cosine_bins.shape}; d shape: {d.shape}")
    print(f"min: {torch.min(cosine_bins)}, max: {torch.max(cosine_bins)}")
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = torch.topk(-torch.square(cosines.unsqueeze(-1) - central_values), k=2)[1][:, :, 1]
    d = torch.abs(cosines - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,
        torch.concatenate([cosine_bins[:, :, :histogram_colon], closest_neighbor.unsqueeze(-1)], dim=-1),
        1 - d
    )

    ################################
    # Azimuth volumes interpolation
    ################################
    central_values, step_size = determine_central_values(start=0., stop=2 * torch.pi, n_bins=azimuth_bins)
    d = torch.abs(v_azimuth - central_values[cosine_bins[:, :, azimuth_colon]]) / step_size

    # Increment histogram bins by 1 - d
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = torch.topk(-torch.square(v_azimuth.unsqueeze(-1) - central_values), k=2)[1][:, :, 1]
    d = torch.abs(v_azimuth - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,
        torch.concatenate(
            [
                cosine_bins[:, :, :azimuth_colon],
                closest_neighbor.unsqueeze(-1),
                cosine_bins[:, :, azimuth_colon + 1:]
            ], dim=-1
        ),
        1 - d
    )

    ##################################
    # Elevation volumes interpolation
    ##################################
    central_values, step_size = determine_central_values(start=0., stop=torch.pi, n_bins=elevation_bins)
    d = torch.abs(v_elevation - central_values[cosine_bins[:, :, elevation_colon]]) / step_size

    # Increment histogram bins by 1 - d
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = torch.topk(-torch.square(v_elevation.unsqueeze(-1) - central_values), k=2)[1][:, :, 1]
    d = torch.abs(v_elevation - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,
        torch.concatenate(
            [
                cosine_bins[:, :, :elevation_colon],
                closest_neighbor.unsqueeze(-1),
                cosine_bins[:, :, elevation_colon + 1:]
            ], dim=-1
        ),
        1 - d
    )

    ###############################
    # Radial volumes interpolation
    ###############################
    central_values, step_size = determine_central_values(start=0., stop=radius, n_bins=radial_bins)
    d = torch.abs(v_radial - central_values[cosine_bins[:, :, radial_colon]]) / step_size

    # Increment histogram bins by 1 - d
    tensor_scatter_nd_add_(histogram, cosine_bins, 1 - d)

    # Determine the closest other bin
    closest_neighbor = torch.topk(-torch.square(v_radial.unsqueeze(-1) - central_values), k=2)[1][:, :, 1]
    d = torch.abs(v_radial - central_values[closest_neighbor]) / step_size

    # Increment neighboring histogram bins by 1 - d
    tensor_scatter_nd_add_(
        histogram,
        torch.concatenate(
            [
                cosine_bins[:, :, :radial_colon],
                closest_neighbor.unsqueeze(-1),
                cosine_bins[:, :, radial_colon + 1:]
            ], dim=-1
        ),
        1 - d
    )

    #########################################
    # Reshape histogram into SHOT-descriptor
    #########################################
    # Reshape histogram into vector
    histogram = histogram.reshape(neighborhood_shape[0], -1)

    # Normalize descriptor to have length 1
    return histogram / torch.linalg.norm(histogram, dim=-1, keepdim=True)