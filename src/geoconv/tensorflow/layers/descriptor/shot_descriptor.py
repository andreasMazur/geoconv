from geoconv.tensorflow.utils.compute_shot_decr import shot_descr
from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf

import tensorflow as tf


class ShotDescriptor(tf.keras.layers.Layer):
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
    ):
        """Initializes the object.

        Parameters
        ----------
        neighbors_for_lrf: int
            The number of neighbors used to construct local reference frames.
        azimuth_bins: int
            The number of azimuth bins.
        elevation_bins: int
            The number of elevation bins.
        radial_bins: int
            The number of radial bins.
        histogram_bins: int
            The number of histogram bins.
        sphere_radius: float
            The sphere radius.
        """
        super().__init__()
        self.neighbors_for_lrf = neighbors_for_lrf
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.radial_bins = radial_bins
        self.histogram_bins = histogram_bins
        self.sphere_radius = sphere_radius

    @tf.function(jit_compile=True)
    def call(self, vertices):
        """Applies the layer to the inputs.

        Parameters
        ----------
        vertices: tf.Tensor
            A tensor of shape 'b x n x 3' containing the 3D vertex coordinates, whereby 'b' represents the number of
            shapes, 'n' the number of vertices per shape.

        Returns
        -------
        tf.Tensor
            A tensor of shape 'b x n x d' containing the SHOT descriptors.
        """
        return tf.map_fn(self.call_helper, vertices)

    @tf.function(jit_compile=True)
    def call_helper(self, vertices):
        """Computes a SHOT descriptor for one point cloud.

        Parameters
        ----------
        vertices: tf.Tensor
            A tensor of shape 'n x 3' containing the 3D vertex coordinates.

        Returns
        -------
        tf.Tensor
            A tensor of shape 'n x d' containing the SHOT descriptors.
        """
        lrfs, neighborhoods, neighborhoods_indices = knn_shot_lrf(
            self.neighbors_for_lrf, vertices[None, ...]
        )
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
                radius=tf.reduce_max(tf.linalg.norm(neighborhoods, axis=-1)),
                azimuth_bins=self.azimuth_bins,
                elevation_bins=self.elevation_bins,
                radial_bins=self.radial_bins,
                histogram_bins=self.histogram_bins,
            )
