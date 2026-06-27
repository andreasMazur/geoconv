from geoconv.tensorflow.layers.convolutions.conv_base import ConvBase

from abc import abstractmethod

import tensorflow as tf
import numpy as np


class ConvIntrinsic(ConvBase):
    """The base class for any non gauge-equivariant surface convolution in GeoConv."""
    def __init__(
        self,
        template_radius,
        rotation_delta,
        output_dim,
        activation,
        include_kernel=True,
        *args,
        **kwargs
    ):
        super().__init__(
            template_radius=template_radius,
            include_kernel=include_kernel,
            activation=activation,
            *args,
            **kwargs
        )
        self.rotation_delta = rotation_delta
        self.output_dim = output_dim

        # Attributes that depend on the data and are set automatically in build
        self._bias = None
        self._template_neighbor_weights = None
        self._template_self_weights = None

    def build(self, inputs):
        """Builds the layer by setting template and bias attributes"""
        super().build(inputs)

        # Init neighbor weights
        self._template_neighbor_weights = self.add_weight(
            name="neighbor_weights",
            shape=(self.output_dim, self.n_radial, self.n_angular, self.feature_dim),
            trainable=True,
        )

        # Init self weights
        self._template_self_weights = self.add_weight(
            name="center_weights",
            shape=(self.output_dim, 1, self.feature_dim),
            trainable=True,
        )

        # Init bias
        self._bias = self.add_weight(
            name="bias", shape=(self.output_dim,), trainable=True
        )

    @tf.function
    def call(self, inputs, orientations=None, **kwargs):
        """Computes intrinsic surface convolution on all vertices of a given mesh.

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (batch_shapes, n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (batch_shapes, n_vertices, n_radial, n_angular, 3, 2).
        orientations: tensorflow.Tensor
            Contains an integer that tells how to rotate the signal-interpolations.

        Returns
        -------
        tf.Tensor
            The geodesic convolution of the template with the signal on the object mesh in every given GPC-system.
            It has size (batch_shapes, vertices, n_rotations, templates)
        """
        mesh_signal, bary_coordinates = inputs

        ####################################################################
        # Fold center - conv_center: (batch_shapes, vertices, 1, templates)
        ####################################################################
        # Weight matrix : (templates, 1, input_dim)
        # Mesh signal   : (batch_shapes, vertices, input_dim)
        # Result        : (batch_shapes, vertices, 1, templates)
        conv_center = tf.einsum("tef,skf->sket", self._template_self_weights, mesh_signal)

        #####################################################################
        # Fold neighbors - conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        #####################################################################
        # Gather signals for patch operator
        # mesh_signal: (batch_shapes, vertices, radial, angular, 3, input_dim)
        # bc_values: (batch_shapes, vertices, radial, angular, 3)
        mesh_signal, bc_values = self._gather_signals(bary_coordinates, mesh_signal)

        # Call patch operator
        interpolations = self._patch_operator(mesh_signal, bc_values)

        # Determine orientations
        if orientations is None:
            # No specific orientations given. Hence, compute for all orientations.
            orientations = tf.range(start=0, limit=self.n_angular, delta=self.rotation_delta)

        def fold_neighbor(o):
            # Weight              : (templates, radial, angular, input_dim)
            # Mesh interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result              : (batch_shapes, vertices, templates)
            return tf.einsum(
                "traf,skraf->skt",
                self._template_neighbor_weights,
                tf.roll(interpolations, shift=o, axis=-2),
            )

        # conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        conv_neighbor = tf.transpose(
            tf.map_fn(fold_neighbor, orientations, fn_output_signature=tf.float32),
            perm=[1, 2, 0, 3],
        )
        return self.activation_fn(conv_center + conv_neighbor + self._bias)

    @abstractmethod
    def define_kernel_values(self, template_matrix):
        """Defines the kernel values for each template vertex.

        Parameters
        ----------
        template_matrix: np.ndarray
            An array of size [n_radial, n_angular, 2], which contains the positions of the template vertices in
            polar coordinates.

        Returns
        -------
        np.ndarray:
            An array of size [n_radial, n_angular, n_radial, n_angular], which contains the interpolation weights for
            the patch operator '[D(x)f](rho_in, theta_in)'
        """
        pass

    def get_config(self):
        """Adds class relevant information to the config-dictionary of the 'ConvBase' class.

        Returns
        -------
        dict:
            The class configuration in the form of a dictionary.
        """
        base_config = super().get_config()
        class_config = {
            "rotation_delta": self.rotation_delta,
            "output_dim": self.output_dim
        }
        base_config.update(class_config)
        return base_config
