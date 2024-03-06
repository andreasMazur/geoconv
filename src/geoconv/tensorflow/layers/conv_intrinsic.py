from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

from abc import ABC, abstractmethod

import tensorflow as tf
import keras


class ConvIntrinsic(ABC, keras.layers.Layer):
    """A metaclass for intrinsic surface convolutions on Riemannian manifolds.

    Attributes
    ----------
    given_name: str

    activation_fn: str
        The activation function to use.
    rotation_delta: int
        The distance between two rotations. If `n` angular coordinates are given in the data, then the default behavior
        is to rotate the template `n` times, i.e. a shift in every angular coordinate of 1 to the next angular
        coordinate.
        If `rotation_delta = 2`, then the shift increases to 2 and the total amount of rotations reduces to
        ceil(n / rotation_delta). This gives a speed-up and saves memory. However, quality of results might worsen.
    amt_templates: int
        The amount of templates to apply during one convolution.
    template_radius: float
        The maximal geodesic extension of the template.
    template_regularizer: str or callable
        A regularizer for the template.
    bias_regularizer: str or callable
        A regularizer for the bias.
    initializer: str or callable
        An initializer for the template and bias.
    include_prior: bool
        Whether to weight the interpolations according to a pre-defined kernel.
    """

    def __init__(self,
                 amt_templates,
                 template_radius,
                 include_prior=True,
                 activation="relu",
                 rotation_delta=1,
                 name=None,
                 template_regularizer=None,
                 bias_regularizer=None,
                 initializer="glorot_uniform"):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()

        self.given_name = name
        self.activation_fn = activation
        self.rotation_delta = rotation_delta
        self.amt_templates = amt_templates
        self.template_radius = template_radius
        self.template_regularizer = template_regularizer
        self.bias_regularizer = bias_regularizer
        self.initializer = initializer
        self.include_prior = include_prior

        # Attributes that depend on the data and are set automatically in build
        self._activation = keras.layers.Activation(self.activation_fn)
        self._bias = None
        self._all_rotations = None
        self._template_size = None  # (#radial, #angular)
        self._template_vertices = None
        self._template_neighbor_weights = None
        self._template_self_weights = None
        self._kernel = None
        self._feature_dim = None

    def get_config(self):
        config = super(ConvIntrinsic, self).get_config()
        config.update(
            {
                "amt_templates": self.amt_templates,
                "template_radius": self.template_radius,
                "include_prior": self.include_prior,
                "activation": self.activation_fn,
                "rotation_delta": self.rotation_delta,
                "name": self.given_name,
                "template_regularizer": self.template_regularizer,
                "bias_regularizer": self.bias_regularizer,
                "initializer": self.initializer
            }
        )
        return config

    def build(self, input_shape):
        """Builds the layer by setting template and bias attributes

        Parameters
        ----------
        input_shape: (tensorflow.TensorShape, tensorflow.TensorShape)
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
        self._template_self_weights = self.add_weight(
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

        # Configure kernel
        self._configure_kernel()

    @tf.function
    def call(self, inputs, orientations=None):
        """Computes intrinsic surface convolution on all vertices of a given mesh.

        Parameters
        ----------
        inputs: (tensorflow.Tensor, tensorflow.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (n_vertices, n_radial, n_angular, 3, 2).
        orientations: tensorflow.Tensor
            Contains an integer that tells how to rotate the data.

        Returns
        -------
        tensorflow.Tensor
            The geodesic convolution of the template with the signal on the object mesh in every given GPC-system.
            It has size (vertices, n_rotations, templates)
        """
        mesh_signal, bary_coordinates = inputs

        ######################################################
        # Fold center - conv_center: (vertices, 1, templates)
        ######################################################
        # Weight matrix : (templates, 1, input_dim)
        # Mesh signal   : (vertices, input_dim)
        # Result        : (vertices, 1, n_templates)
        conv_center = tf.einsum("tef,kf->ket", self._template_self_weights, mesh_signal)

        #####################################################################
        # Fold neighbors - conv_neighbor: (vertices, n_rotations, templates)
        #####################################################################
        # Call patch operator
        interpolations = self._patch_operator(mesh_signal, bary_coordinates)
        # Determine orientations
        if orientations is None:
            # No specific orientations given. Hence, compute for all orientations.
            orientations = tf.range(start=0, limit=self._all_rotations, delta=self.rotation_delta)

        def fold_neighbor(o):
            # Weight              : (templates, radial, angular, input_dim)
            # Mesh interpolations : (vertices, radial, angular, input_dim)
            # Result              : (vertices, templates)
            return tf.einsum(
                "traf,kraf->kt",
                self._template_neighbor_weights,
                tf.roll(interpolations, shift=o, axis=2)
            )

        # conv_neighbor: (vertices, n_rotations, templates)
        conv_neighbor = tf.transpose(
            tf.map_fn(fold_neighbor, orientations, fn_output_signature=tf.float32), perm=[1, 0, 2]
        )
        return self._activation(conv_center + conv_neighbor + self._bias)

    @tf.function
    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Interpolates and weights mesh signal

        Parameters
        ----------
        mesh_signal: tensorflow.Tensor
            The signal values at the template vertices
        barycentric_coordinates: tensorflow.Tensor
            The barycentric coordinates for the template vertices

        Returns
        -------
        tensorflow.Tensor:
            Weighted and interpolated mesh signals
        """
        interpolations = self._signal_retrieval(mesh_signal, barycentric_coordinates)

        if self.include_prior:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (vertices, radial, angular, input_dim)
            # Result         : (vertices, radial, angular, input_dim)
            return tf.einsum("raxy,kxyf->kraf", self._kernel, interpolations)
        else:
            return interpolations

    @tf.function
    def _signal_retrieval(self, mesh_signal, barycentric_coordinates):
        """Interpolates signals at template vertices

        Parameters
        ----------
        mesh_signal: tensorflow.Tensor
            The signal values at the template vertices
        barycentric_coordinates: tensorflow.Tensor
            The barycentric coordinates for the template vertices

        Returns
        -------
        tensorflow.Tensor:
            Interpolation values for the template vertices
        """
        vertex_indices = tf.reshape(
            tf.cast(barycentric_coordinates[:, :, :, :, 0], tf.int32), (-1, 1)
        )
        mesh_signal = tf.reshape(
            tf.gather_nd(mesh_signal, vertex_indices),
            (-1, self._template_size[0], self._template_size[1], 3, self._feature_dim)
        )
        # (vertices, n_radial, n_angular, input_dim)
        return tf.math.reduce_sum(
            tf.expand_dims(barycentric_coordinates[:, :, :, :, 1], axis=-1) * mesh_signal, axis=-2
        )

    def _configure_kernel(self):
        """Defines all necessary interpolation coefficient matrices for the patch operator."""
        self._kernel = tf.cast(
            self.define_kernel_values(self._template_vertices.numpy()), tf.float32
        )

    @abstractmethod
    def define_kernel_values(self, template_matrix):
        """Defines the kernel values for each template vertex.

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
