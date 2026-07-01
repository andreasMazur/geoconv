from geoconv.pytorch.utils.compute_shot_descr import shot_descr
from geoconv.pytorch.utils.compute_shot_lrf import knn_shot_lrf

from torch import nn

import torch


class ShotDescriptor(nn.Module):
    """This class implements a layer on top of the computation of SHOT-descriptors.

    Attributes
    ----------
    neighbors_for_lrf: int
        The amount of neighbors to consider while computing the LRFs.
    azimuth_bins: int
        The amount of azimuths for the SHOT-descriptors.
    elevation_bins: int
        The amount of elevation for the SHOT-descriptors.
    radial_bins: int
        The amount of radial bins for the SHOT-descriptors.
    histogram_bins: int
        The amount of bins in a histogram for a cell in the SHOT-descriptor.
    sphere_radius: float
        The radius of the sphere for the SHOT-descriptor.
    """
    def __init__(
        self,
        neighbors_for_lrf=16,
        azimuth_bins=8,
        elevation_bins=2,
        radial_bins=2,
        histogram_bins=11,
        sphere_radius=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.neighbors_for_lrf = neighbors_for_lrf
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.radial_bins = radial_bins
        self.histogram_bins = histogram_bins
        self.sphere_radius = sphere_radius

    def forward(self, vertices):
        return torch.stack([self.forward_helper(shape) for shape in vertices], dim=-1)

    def forward_helper(self, vertices):
        lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(self.neighbors_for_lrf, vertices[None, ...])
        lrfs, neighborhoods, neighborhoods_indices = lrfs[0], neighborhoods[0], neighborhoods_indices[0]
        if self.sphere_radius > 0.0:
            return shot_descr(
                neighborhoods=neighborhoods,
                normals=lrfs[..., 0],
                neighborhood_indices=neighborhoods_indices,
                radius=self.sphere_radius,
                azimuth_bins=self.azimuth_bins,
                elevation_bins=self.elevation_bins,
                radial_bins=self.radial_bins,
                histogram_bins=self.histogram_bins,
            )
        else:
            return shot_descr(
                neighborhoods=neighborhoods,
                normals=lrfs[..., 0],
                neighborhood_indices=neighborhoods_indices,
                radius=float(torch.amax(torch.linalg.norm(neighborhoods, dim=-1))),
                azimuth_bins=self.azimuth_bins,
                elevation_bins=self.elevation_bins,
                radial_bins=self.radial_bins,
                histogram_bins=self.histogram_bins,
            )
