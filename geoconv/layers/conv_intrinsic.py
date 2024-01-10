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
        self._template_weights = None
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
        self._template_weights = self.add_weight(
            name="conv_intrinsic_template",
            shape=(self.amt_templates, 1, signal_shape[1], self._template_size[0] * self._template_size[1]),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.template_regularizer
        )

        self._bias = self.add_weight(
            name="conv_intrinsic_bias",
            shape=(self.amt_templates, 1),
            initializer=self.initializer,
            trainable=True,
            regularizer=self.bias_regularizer
        )

        # Configure patch operator
        self._configure_patch_operator()

    @tf.function
    def call(self, inputs, orientation=tf.constant(-1)):
        """Computes intrinsic surface convolution for multiple given GPC-systems

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (n_vertices, n_radial, n_angular, 3, 2).
        orientation: tf.Tensor
            Contains an integer that tells how to rotate the data.

        Returns
        -------
        tf.Tensor
            The geodesic convolution of the template with the signal on the object mesh in every given GPC-system.
            It has size (n_batch, n_vertices, feature_dim)
        """
        mesh_signal, bary_coordinates = inputs
        mesh_signal = self._patch_operator(mesh_signal, bary_coordinates)
        idx = tf.constant(0)
        new_signal = tf.TensorArray(
            tf.float32,
            size=self.splits,
            dynamic_size=False,
            clear_after_read=True,
            tensor_array_name="outer_ta",
            name="call_ta"
        )
        if orientation > tf.constant(-1):
            for interpolations in tf.split(mesh_signal, self.splits):
                new_signal = new_signal.write(idx, self._select_orientations(interpolations, orientation))
                idx = idx + tf.constant(1)
            new_signal = new_signal.concat()
        else:
            for interpolations in tf.split(mesh_signal, self.splits):
                new_signal = new_signal.write(idx, self._select_orientations(interpolations))
                idx = idx + tf.constant(1)
            new_signal = new_signal.concat()
        return new_signal

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
    def _select_orientations(self, interpolations, orientation=tf.constant(-1)):
        idx = tf.constant(0)
        all_rotations = tf.range(start=0, limit=self._all_rotations, delta=self.rotation_delta)
        size = tf.shape(all_rotations)[0]
        new_signal = tf.TensorArray(
            tf.float32,
            size=size,
            dynamic_size=False,
            clear_after_read=True,
            tensor_array_name="inner_ta",
            name="rotation_ta"
        )
        if orientation > tf.constant(-1):
            # (n_templates, subset)
            new_signal = self._fold(interpolations, orientation)
            # (subset, n_templates)
            new_signal = tf.transpose(new_signal, perm=[1, 0])
            # add missing dimension: (n_vertices, 1, n_templates)
            return tf.expand_dims(new_signal, axis=1)
        else:
            # Iterate over rotations to economize memory usage
            for rot in all_rotations:
                rotated_interpolations = self._fold(interpolations, rot)
                # Store result:
                new_signal = new_signal.write(idx, rotated_interpolations)
                idx = idx + tf.constant(1)
            # Stack results for all rotations: (n_rotations, n_templates, subset)
            new_signal = new_signal.stack()
            # Transpose for AMP: (subset, n_rotations, n_templates)
            return tf.transpose(new_signal, perm=[2, 0, 1])

    @tf.function
    def _fold(self, interpolations, orientation):
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
        # Rotation: (subset, n_radial, n_angular, input_dim)
        rotated_interpolations = tf.roll(interpolations, shift=orientation, axis=2)
        # Reshape for conv: (subset, n_radial * n_angular, input_dim)
        rotated_interpolations = tf.reshape(
            rotated_interpolations, (-1, self._template_size[0] * self._template_size[1], self._feature_dim)
        )
        # Compute conv:
        # Interpolated signals: (              subset, n_radial * n_angular,            input_dim)
        # Weight matrix       : (n_templates,       1,            input_dim, n_radial * n_angular)
        # Bias                : (n_templates,       1,                                           )
        # Result              : (n_templates,  subset                                            )
        return self._activation(
            tf.einsum("ijk,xykj->xi", rotated_interpolations, self._template_weights) + self._bias
        )

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
