from geoconv.preprocessing.barycentric_coordinates import create_template_matrix
from geoconv.pytorch.layers.normalize_point_cloud import NormalizePointCloud
from geoconv.pytorch.utils.compute_shot_lrf import logarithmic_map, knn_shot_lrf

import torch
import sys

from typing import Tuple

import warnings

@torch.jit.script
def compute_det(batched_matrices: torch.Tensor) -> torch.Tensor:
    """Computes the determinant of a batch of matrices.

    Parameters
    ----------
    batched_matrices: torch.Tensor
        A 3D-tensor of shape (batch_size, 3, 3) that contains the matrices for which the determinant should be computed.

    Returns
    -------
    torch.Tensor:
        A 1D-tensor of shape (batch_size,) that contains the determinants of the input matrices.
    """
    return torch.det(batched_matrices)

@torch.jit.script
def sort_angles(angles: torch.Tensor) -> torch.Tensor:
    # Create indices
    indices = torch.broadcast_to(torch.arange(3, device=angles.device)[None, :], angles.shape)

    # Initially compare x1 and x2
    smaller_mask = angles[:, 0] > angles[:, 1]
    smaller = torch.where(smaller_mask, indices[:, 1], indices[:, 0])
    larger  = torch.where(smaller_mask, indices[:, 0], indices[:, 1])

    # Find the largest by comparing 'larger' and x3
    largest = torch.where(angles[:, 2] < angles[torch.arange(angles.shape[0], device=angles.device), larger], larger, indices[:, 2])

    # Find the smallest by comparing 'smaller' and x3
    smaller = torch.where(angles[:, 2] > angles[torch.arange(angles.shape[0], device=angles.device), smaller], smaller, indices[:, 2])

    return torch.stack([smaller, 3 - (smaller + largest), largest], dim=-1)

@torch.jit.script
def sort_triangles_ccw(triangles : torch.Tensor) -> torch.Tensor:
    centroid = torch.mean(triangles, dim=1, keepdim=True)
    angles = torch.atan2(triangles[..., 1] - centroid[..., 1], triangles[..., 0] - centroid[..., 0])
    sorted_indices = sort_angles(angles)
    print(f"centroid: {centroid.shape}; angles: {angles.shape}; sorted_indices: {sorted_indices.shape}")
    return triangles.gather(1, sorted_indices.unsqueeze(-1).expand(-1,-1,2))

@torch.jit.script
def delaunay_condition_check(triangles : torch.Tensor, projections : torch.Tensor) -> torch.Tensor:
    # Delaunay condition-check requires counter-clock-wise (ccw) sorted triangles
    triangles = sort_triangles_ccw(triangles.reshape(-1, 3, 2)).reshape(triangles.shape) 

    # triangles must be rotated counterclockwise
    # `column_1_2`: (n_vertices, n_neighbors, `n_neighbors over 3`)
    # `column_1_2[..., 0]` -> x-coodrinate difference to triangle vertices from selected neighbor
    # `column_1_2[..., 1]` -> y-coodrinate difference to triangle vertices from selected neighbor
    column_1_2 = triangles[:, None, ...] - projections[..., None, None, :]

    # `delaunay_check_matrix`: (n_vertices, n_neighbors, `n_neighbors over 3`)
    # True if projection `i` is outside of circumcircle of triangle `j` in neighborhood `k`
    delaunay_check_matrix = (compute_det(
        torch.stack([
            column_1_2[..., 0],
            column_1_2[..., 1],
            torch.square(column_1_2[..., 0]) + torch.square(column_1_2[..., 1])
        ], dim=-1)
    ) > 0.).to(torch.int32)

    # `delaunay_check_matrix.sum(dim=1)`: (n_vertices, `n_neighbors over 3`)
    return delaunay_check_matrix.sum(dim=1) > 0

@torch.jit.script
def create_all_triangles(projections : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates all triangles for a given set of projections."""
    p_shape = projections.shape

    # Create all possible (i, j, k) index triplets
    I, J, K = torch.meshgrid(torch.arange(p_shape[-2], device=projections.device), torch.arange(p_shape[-2], device=projections.device), torch.arange(p_shape[-2], device=projections.device), indexing="ij")

    # Filter out invalid combinations (where i < j < k)
    triangle_indices = torch.stack(torch.where(torch.logical_and(I < J, J < K)), dim=-1)
    triangles = projections[:, triangle_indices.long().t()].transpose(1,2)

    return triangles, triangle_indices

@torch.jit.script
def compute_interpolation_coefficients(triangles : torch.Tensor, template : torch.Tensor) -> torch.Tensor:
    v0 = triangles[..., 2, :] - triangles[..., 0, :]
    v1 = triangles[..., 1, :] - triangles[..., 0, :]
    v2 = template[None, ..., None, :] - triangles[:, None, None, :, 0, :]

    dot00 = torch.einsum("ijk,ijk->ij", v0, v0)[:, None, None, :]
    dot01 = torch.einsum("ijk,ijk->ij", v0, v1)[:, None, None, :]
    dot02 = torch.einsum("ijk,irajk->iraj", v0, v2)
    dot11 = torch.einsum("ijk,ijk->ij", v1, v1)[:, None, None, :]
    dot12 = torch.einsum("ijk,irajk->iraj", v1, v2)

    denominator = 1. / (dot00 * dot11 - dot01 * dot01)
    point_2_weight = (dot11 * dot02 - dot01 * dot12) * denominator
    point_1_weight = (dot00 * dot12 - dot01 * dot02) * denominator
    point_0_weight = 1. - point_1_weight - point_2_weight

    bc_coordinates = torch.stack([point_0_weight, point_1_weight, point_2_weight], dim=-1)

    # Set NAN-values to -1. so they get filtered out by BC-condition (np.inf would also work)
    nan_mask = torch.isnan(bc_coordinates) 
    bc_coordinates[nan_mask] = torch.tensor([-1.], dtype=bc_coordinates.dtype, device=bc_coordinates.device).repeat(torch.sum(nan_mask))
    return bc_coordinates

@torch.jit.script
def compute_interpolation_weights(template : torch.Tensor, projections : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # `triangles`: (n_vertices, `n_neighbors over 3`, 3, 2)
    # `triangle_indices`: (`n_neighbors over 3`, 3)
    triangles, triangle_indices = create_all_triangles(projections)

    # Use Delaunay condition to remove triangle-pairs that could not co-exists
    # `delauanay_condition[x, y] = True` if `triangles[x, y]` meets Delauanay condition
    # `delauanay_condition`: (n_vertices, `n_neighbors over 3`)
    delauanay_condition = delaunay_condition_check(triangles, projections)

    # `barycentric_coordinates`: (n_vertices, n_radial, n_angular, `n_neighbors over 3`, 3)
    # TODO: Mostly close to tf, but some entry-wise differences. Check thoroughly
    barycentric_coordinates = compute_interpolation_coefficients(triangles, template)

    # `mask`: (n_vertices, n_radial, n_angular, `n_neighbors over 3`)
    bc_condition = torch.any(
        torch.logical_or(barycentric_coordinates > 1., barycentric_coordinates < 0.),
        dim=-1
    )
    mask = torch.logical_or(delauanay_condition[:, None, None, :], bc_condition)

    # `tri_distances`: (n_vertices, n_radial, n_angular, `n_neighbors over 3`)
    tri_distances = torch.sum(
        torch.linalg.norm(
            triangles[:, None, None, ...] - template[None, :, :, None, None, :], dim=-1
        ), dim=-1
    )

    # Set triangle distance to infinity where conditions aren't met
    tri_distances[mask] = torch.tensor([torch.inf]).to(
        device=tri_distances.device, dtype=tri_distances.dtype
    ).repeat(torch.sum(mask))
    closest_triangles = torch.argmin(tri_distances, dim=-1)

    # Select bc of closest possible triangle
    selected_bc = torch.gather(
        barycentric_coordinates,
        dim=3,
        index=closest_triangles[..., None, None].expand(-1, -1, -1, -1, 3)
    ).squeeze(-2)

    expanded_closest_triangles = closest_triangles.view(-1, triangle_indices.shape[0], 1).expand(-1, -1, 3)
    expanded_indices = triangle_indices.unsqueeze(0).expand(expanded_closest_triangles.shape[0], -1, -1)

    correction_mask = torch.all(mask, dim=-1)

    selected_indices = torch.gather(
        expanded_indices,
        dim=1,
        index=expanded_closest_triangles
    ).view(selected_bc.shape[0], selected_bc.shape[1], selected_bc.shape[2], 3).to(torch.int32)

    # Might happen that no triangles fit for a template vertex. Set those interpolation coefficients to zero.
    selected_bc[correction_mask] = 0.
    selected_indices[correction_mask] = 0

    return selected_bc, selected_indices


@torch.jit.script
def compute_bc(template : torch.Tensor, projections : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes barycentric coordinates for a given template in given projections.
    
    Parameters
    ----------
    template: torch.Tensor
        A 3D-tensor of shape (n_radial, n_angluar, 2) that contains 2D cartesian coordinates for template vertices.
    projections: torch.Tensor
        A 3D-tensor of shape (vertices, n_neighbors, 2) that contains all projected neighborhoods in 2D catesian
        coordinates. I.e. `projections[i, j]` contains 2D coordinates of vertex `j` in neighborhood `i`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]:
        A 4D-tensor of shape (vertices, n_neighbors, n_radial, n_angular) that contains the barycentric coordinates, i.e.,
        interpolation coefficients, for all template vertices within each projected neighborhood. Additionally,
        another 4D-tensor of shape (vertices, n_radial, n_angular, 3) that contains the vertex indices of the closest
        projected vertices to the template vertices in each neighborhood is returned.
    """
    template = template.to(torch.float64)
    projections = projections.to(torch.float64)
    interpolation_weights, interpolation_indices = compute_interpolation_weights(template, projections)

    return interpolation_weights, interpolation_indices

class BarycentricCoordinates(torch.jit.ScriptModule):
    """A parameter-free neural network layer that approximates barycentric coordinates (BC).
    
    Attributes
    ----------
    n_radial: int
        The number of radial coordinates of the template for which BC shall be computed.
    n_angular: int
        The number of angular coordinates of the template for which BC shall be computed.
    projection_neighbors: int
        The number of neighbors that shall be projected. Has to e smaller or equal than `neighbors_for_lrf`.
        These are also used to determine the template radius.
    neighbors_for_lrf: int
        The number of neighbors that shall be used to compute the normal vectors of the local reference frames.
    """
    def __init__(self, n_radial : int, n_angular : int, projection_neighbors : int = 8, neighbors_for_lrf : int = 16):
        super().__init__()
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.projection_neighbors = projection_neighbors
        self.neighbors_for_lrf = neighbors_for_lrf

        if projection_neighbors > neighbors_for_lrf:
            warnings.warn(
                f"### You wanted to use {projection_neighbors} projections but created LRFs with only "
                f"{neighbors_for_lrf} vertices. Therefore, this BC-layer will only use {neighbors_for_lrf} "
                "projections. ###"
            )

    def adapt(self,
              data : torch.utils.data.Dataset = None,
              template_scale : float = None,
              template_radius : float = None,
              with_normalization : bool = True,
              exp_lambda : float = 1.0,
              shift_angular : bool = True) -> float:
        """Sets the template radius to a given or the average neighborhood radius scaled by used defined coefficient.

        Parameters
        ----------
        data: torch.utils.data.Dataset
            The training data which is used to compute the template radius.
        template_scale: float
            The scaling factor to multiply on the template.
        template_radius: float
            The template radius to use to initialize the template.
        with_normalization: bool
            Whether to normalize the point-cloud before projection.
        exp_lambda: float
            Whether to sample more points closer to the origin than farther out. This lambda determines the strength
            of how non-uniform to sample.
        shift_angular: bool
            Whether to add half of angular step to every second row of template vertices.

        Returns
        -------
        float:
            The final template radius.
        """
        if template_radius is None:
            assert data is not None, "If 'template_radius' is not given, 'data' has to be provided."
            assert template_scale is not None, "If 'template_radius' is not given, 'template_scale' has to be provided."
        
            # If no template radius is given, compute the average neighborhood radius
            normalization_layer = NormalizePointCloud()
            avg_radius, vertices_count = 0, 0
            for idx, (vertices, _) in enumerate(data):
                assert vertices.shape[0] == 1, "Batch-size has to be 1 for BC-layer adaptation."

                sys.stdout.write(f"\rCurrently at point-cloud {idx}.")
                # 0.) Point-cloud normalization
                if with_normalization:
                    vertices = normalization_layer(vertices)
                
                # 1.) Compute projections
                projections, _ = self.project(vertices[0])

                # 2.) Use length of the farthest projection as radius
                radii = torch.max(torch.linalg.norm(projections, dim=-1), dim=-1)

                # 3.) Add all radii
                avg_radius += torch.sum(radii)

                # 4.) Remember the number of collected radii for averaging
                vertices_count += radii.shape[0]
            avg_radius /= vertices_count
            template_radius = avg_radius * template_scale
        
        # Initialize template
        self.template = create_template_matrix(
            n_radial=self.n_radial,
            n_angular=self.n_angular,
            radius=template_radius,
            in_cart=True,
            exp_lambda=exp_lambda,
            shift_angular=shift_angular
        ).to(torch.float32, device=self.device)

        # Return used template radius
        return template_radius

    def project(self, vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get local reference frames
        # 'lrfs': (vertices, 3, 3)
        lrfs, neighborhoods, neighborhood_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices)

        # Project neighborhoods into their lrfs using the logarithmic map
        # 'projections': (vertices, n_neighbors, 2)
        projections = logarithmic_map(lrfs, neighborhoods)

        return projections[:, :self.projection_neighbors, :], neighborhood_indices[:, :self.projection_neighbors]


# @torch.jit.script
# def call_helper(self, )