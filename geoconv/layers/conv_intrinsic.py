from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

from tensorflow import keras
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np


class ConvIntrinsic(ABC, keras.layers.Layer):
    """A metaclass for intrinsic surface convolutions on Riemannian manifolds.

    Attributes
    ----------
    amt_templates: int
        The amount of templates to apply during one convolution.
    activation_fn: str
        The activation function to use.
    rotation_delta: int
        The distance between two rotations. If `n` angular coordinates are given in the data, then the default behavior
        is to rotate the template `n` times, i.e. a shift in every angular coordinate of 1 to the next angular
        coordinate.
        If `rotation_delta = 2`, then the shift increases to 2 and the total amount of rotations reduces to
        ceil(n / rotation_delta). This gives a speed-up and saves memory. However, quality of results might worsen.
    splits: int
        The 'splits'-parameter determines into how many chunks the mesh signal is split. Each chunk will be folded
        iteratively to save memory. That is, fewer splits allow a faster convolution. More splits allow reduced memory
        usage. Careful: 'splits' has to divide the amount of vertices in the input mesh. Also, using many splits might
        cause larger memory fragmentation.
    include_prior: bool
        Determines whether to include prior. If 'False', computation is faster.
    """

    def __init__(self,
                 amt_templates,
                 template_radius,
                 activation="relu",
                 rotation_delta=1,
                 splits=1,
                 include_prior=True,
                 name=None,
                 template_regularizer=None,
                 bias_regularizer=None,
                 initializer="glorot_uniform"):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        self.activation_fn = activation
        self.rotation_delta = rotation_delta
        self.amt_templates = amt_templates
        self.template_radius = template_radius
        self.template_regularizer = template_regularizer
        self.bias_regularizer = bias_regularizer
        self.initializer = initializer
        self.splits = splits
        self.include_prior = include_prior

        # Attributes that depend on the data and are set automatically in build
        self._activation = keras.layers.Activation(self.activation_fn)
        self._bias = None
        self._all_rotations = None
        self._template_size = None  # (#radial, #angular)
        self._template_vertices = None
        self._template_neighbor_weights = None
        self._template_center_weights = None
        self._interpolation_coefficients = None
        self._feature_dim = None

    def get_config(self):
        config = super(ConvIntrinsic, self).get_config()
        config.update(
            {
                "amt_template": self.amt_templates,
                "template_radius": self.template_radius,
                "activation_fn": self.activation_fn,
                "rotation_delta": self.rotation_delta,
                "splits": self.splits,
                "name": self.name,
                "template_regularizer": self.template_regularizer,
                "bias_regularizer": self.bias_regularizer,
                "initializer": self.initializer,
                "include_prior": self.include_prior
            }
        )
        return config

    def build(self, input_shape):
        """Builds the layer by setting template and bias attributes

        Parameters
        ----------
        input_shape: (tf.TensorShape, tf.TensorShape)
            The shape of the signal and the shape of the barycentric coordinates.
        """
        signal_shape, barycentric_shape = input_shape

        # Configure template
        self._template_size = (barycentric_shape[1], barycentric_shape[2])
        self._all_rotations = self._template_size[1]
        self._template_vertices = tf.constant(
            create_template_matrix(self._template_size[0], self._template_size[1], radius=self.template_radius)
        )
        self._feature_dim = signal_shape[-1]

        # Configure trainable weights
        self._template_neighbor_weights = self.add_weight(
            name="neighbor_weights",
            shape=(self.amt_templates, self._template_size[0], self._template_size[1], signal_shape[1]),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.template_regularizer
        )
        self._template_center_weights = self.add_weight(
            name="center_weights",
            shape=(self.amt_templates, 1, signal_shape[1]),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.template_regularizer
        )
        self._bias = self.add_weight(
            name="conv_intrinsic_bias",
            shape=(self.amt_templates,),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.bias_regularizer
        )

        # Configure patch operator
        self._configure_patch_operator()

    @tf.function
    def call(self, inputs, orientations=tf.constant([], dtype=tf.int32)):
        """Computes intrinsic surface convolution for multiple given GPC-systems

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (n_vertices, n_radial, n_angular, 3, 2).
        orientations: tf.Tensor
            Contains an integer that tells how to rotate the data.

        Returns
        -------
        tf.Tensor
            The geodesic convolution of the template with the signal on the object mesh in every given GPC-system.
            It has size (n_batch, n_vertices, feature_dim)
        """
        mesh_signal, bary_coordinates = inputs

        # Call patch operator
        interpolations = self._patch_operator(mesh_signal, bary_coordinates)

        # Batch input
        mesh_signal = tf.stack(tf.split(mesh_signal, self.splits))
        interpolations = tf.stack(tf.split(interpolations, self.splits))

        # Fold center features
        conv_center = tf.reshape(tf.map_fn(self._fold_center, mesh_signal), (-1, 1, self.amt_templates))

        # Fold neighbor features
        if tf.equal(tf.size(orientations), 0):
            # No specific orientations given. Hence, compute for all orientations.
            orientations = tf.range(start=0, limit=self._all_rotations, delta=self.rotation_delta)

        batched_folding = lambda batch: self._fold_neighbors(batch, orientations)
        conv_neighbor = tf.reshape(
            tf.map_fn(batched_folding, interpolations), (-1, tf.shape(orientations)[0], self.amt_templates)
        )
        return self._activation(conv_center + conv_neighbor)

    @tf.function
    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Implements the patch operator

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The signal values at the template vertices
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices

        Returns
        -------
        tf.Tensor:
            Interpolation values for the template vertices
        """
        ############################################
        # Signal-interpolation at template vertices
        ############################################
        mesh_signal = tf.reshape(
            tf.gather(mesh_signal, tf.reshape(tf.cast(barycentric_coordinates[:, :, :, :, 0], tf.int32), (-1,))),
            (-1, self._template_size[0], self._template_size[1], 3, self._feature_dim)
        )
        # (subset, n_radial, n_angular, input_dim)
        mesh_signal = tf.math.reduce_sum(
            tf.expand_dims(barycentric_coordinates[:, :, :, :, 1], axis=-1) * mesh_signal, axis=-2
        )
        if not self.include_prior:
            return mesh_signal

        ##################
        # Including prior
        ##################
        # (subset, n_radial * n_angular, input_dim)
        mesh_signal = tf.reshape(mesh_signal, (-1, self._template_size[0] * self._template_size[1], self._feature_dim))

        # (subset, input_dim, n_radial * n_angular)
        mesh_signal = tf.transpose(mesh_signal, perm=[0, 2, 1])

        # (subset, 1, 1, input_dim, n_radial * n_angular)
        mesh_signal = tf.expand_dims(tf.expand_dims(mesh_signal, axis=1), axis=1)

        # (subset,       1,          1, input_dim, n_radial * n_angular)
        # (        n_radial, n_angular,            n_radial * n_angular)
        # (subset, n_radial, n_angular, input_dim                      )
        return tf.linalg.matvec(mesh_signal, self._interpolation_coefficients)

    @tf.function
    def _fold_center(self, mesh_signal):
        """Folds the features in the center of the template

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The mesh signal at the mesh vertices.

        Returns
        -------
        tf.Tensor:
            New mesh signal at the mesh vertices.
        """
        # Mesh signal   : (subset, input_dim)
        # Weight matrix : (n_templates, 1, input_dim)
        # Result        : (subset, 1, n_templates)
        return tf.einsum("si,tji->sjt", mesh_signal, self._template_center_weights)

    @tf.function
    def _fold_neighbors(self, interpolations, considered_rotations):
        """Folds template vertex signal with the template weights

        Parameters
        ----------
        interpolations: tf.Tensor
            The according to a given weighting function weighted interpolations at the template vertices.
            Shape: (subset, n_radial, n_angular, input_dim)
        orientation: tf.Tensor
            Contains an integer that tells how to rotate the data.

        Returns
        -------
        tf.Tensor:
            The new mesh signal after the convolution. Shape: (subset, n_rotations, n_templates)
        """
        # n_rotations = ceil(self.all_rotations / self.rotation_delta)
        all_rotations_fn = lambda rot: tf.roll(interpolations, shift=rot, axis=2)
        # (n_rotations, subset, n_radial, n_angular, input_dim)
        interpolations = tf.map_fn(all_rotations_fn, considered_rotations, fn_output_signature=tf.float32)

        # Interpolated signals: (n_rotations, subset,                      n_radial, n_angular, input_dim)
        # Weight matrix       : (                     n_templates,         n_radial, n_angular, input_dim)
        # Bias                : (                     n_templates                                        )
        # Result              : (subset, n_rotations, n_templates)
        return tf.einsum("rsijk,tijk->srt", interpolations, self._template_neighbor_weights) + self._bias

    def _configure_patch_operator(self):
        """Defines all necessary interpolation coefficient matrices for the patch operator."""

        self._interpolation_coefficients = tf.cast(
            self.define_interpolation_coefficients(self._template_vertices.numpy()), tf.float32
        )
        self._interpolation_coefficients = tf.reshape(
            self._interpolation_coefficients,
            (self._template_size[0], self._template_size[1], self._template_size[0] * self._template_size[1])
        )

    @abstractmethod
    def define_interpolation_coefficients(self, template_matrix):
        """Defines the interpolation coefficients for each template vertex.

        Parameters
        ----------
        template_matrix: np.ndarray
            An array of size [n_radial, n_angular, 2], which contains the positions of the template vertices in
            cartesian coordinates.

        Returns
        -------
        np.ndarray:
            An array of size [n_radial, n_angular, n_radial, n_angular], which contains the interpolation weights for
            the patch operator '[D(x)f](rho_in, theta_in)'
        """
        pass
