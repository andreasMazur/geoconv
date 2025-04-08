from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

from abc import ABC, abstractmethod

import tensorflow as tf


class ConvIntrinsic(ABC, tf.keras.layers.Layer):
    """A metaclass for intrinsic surface convolutions.

    Attributes
    ----------
    activation: str
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
    include_prior: bool
        Whether to weight the interpolations according to a pre-defined kernel.
    """

    def __init__(
        self,
        amt_templates,
        template_radius,
        include_prior=True,
        activation="relu",
        rotation_delta=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.amt_templates = amt_templates
        self.template_radius = template_radius
        self.include_prior = include_prior
        self.activation = activation
        self.rotation_delta = rotation_delta

        # Attributes that depend on the data and are set automatically in build
        self._activation = tf.keras.layers.Activation(self.activation)
        self._bias = None
        self._all_rotations = None
        self._template_size = None  # (#radial, #angular)
        self._template_vertices = None
        self._template_neighbor_weights = None
        self._template_self_weights = None
        self._kernel = None
        self._feature_dim = None
        self._input_shape = None

    def get_config(self):
        """Get the configuration dictionary.

        Returns
        -------
        dict:
            The configuration dictionary.
        """
        config = super(ConvIntrinsic, self).get_config()
        config.update(
            {
                "amt_templates": self.amt_templates,
                "template_radius": self.template_radius,
                "include_prior": self.include_prior,
                "activation": self.activation,
                "rotation_delta": self.rotation_delta,
                "input_shape": self._input_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Re-instantiates the layer from the config dictionary.

        Parameters
        ----------
        config: dict
            The configuration dictionary.

        Returns
        -------
        ConvIntrinsic:
            The layer.
        """
        model = cls(**config)
        model.build(config["input_shape"])
        return model

    def build(self, input_shape):
        """Builds the layer by setting template and bias attributes

        Parameters
        ----------
        input_shape: (tf.TensorShape, tf.TensorShape)
            The shape of the signal and the shape of the barycentric coordinates.
        """
        # Remember input-shape
        self._input_shape = input_shape
        signal_shape, barycentric_shape = self._input_shape

        # Configure layer-attributes that depend on the input shapes
        self._template_size = (barycentric_shape[-4], barycentric_shape[-3])
        self._template_vertices = tf.constant(
            create_template_matrix(
                self._template_size[0],
                self._template_size[1],
                radius=self.template_radius,
            )
        )
        self._all_rotations = self._template_size[1]
        self._feature_dim = signal_shape[-1]

        # Configure trainable weights
        self._template_neighbor_weights = self.add_weight(
            name="neighbor_weights",
            shape=(
                self.amt_templates,
                self._template_size[0],
                self._template_size[1],
                signal_shape[-1],
            ),
            trainable=True,
        )
        self._template_self_weights = self.add_weight(
            name="center_weights",
            shape=(self.amt_templates, 1, signal_shape[-1]),
            trainable=True,
        )
        self._bias = self.add_weight(
            name="bias", shape=(self.amt_templates,), trainable=True
        )

        # Configure kernel
        self._configure_kernel()

    @tf.function
    def call(self, inputs, orientations=None, **kwargs):
        """Computes intrinsic surface convolution on all vertices of a given mesh.

        Parameters
        ----------
        **kwargs
        inputs: (tensorflow.Tensor, tensorflow.Tensor)
            The first tensor represents the signal defined on the manifold. It has size
            (batch_shapes, n_vertices, feature_dim). The second tensor represents the barycentric coordinates. It has
            size (batch_shapes, n_vertices, n_radial, n_angular, 3, 2).
        orientations: tensorflow.Tensor
            Contains an integer that tells how to rotate the signal-interpolations.

        Returns
        -------
        tensorflow.Tensor
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
        conv_center = tf.einsum(
            "tef,skf->sket", self._template_self_weights, mesh_signal
        )

        #####################################################################
        # Fold neighbors - conv_neighbor: (batch_shapes, vertices, n_rotations, templates)
        #####################################################################
        # Call patch operator
        interpolations = self._patch_operator(mesh_signal, bary_coordinates)
        # Determine orientations
        if orientations is None:
            # No specific orientations given. Hence, compute for all orientations.
            orientations = tf.range(
                start=0, limit=self._all_rotations, delta=self.rotation_delta
            )

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
        # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
        interpolations = self._signal_pullback(mesh_signal, barycentric_coordinates)

        if self.include_prior:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result         : (batch_shapes, vertices, radial, angular, input_dim)
            return tf.einsum("raxy,skxyf->skraf", self._kernel, interpolations)
        else:
            return interpolations

    @tf.function
    def _signal_pullback(self, mesh_signal, barycentric_coordinates):
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
        # Get vertex indices from BC-tensor
        vertex_indices = tf.reshape(
            tf.cast(barycentric_coordinates[:, :, :, :, :, 0], tf.int32),
            (-1, tf.reduce_prod(tf.shape(barycentric_coordinates)[1:-1]), 1),
        )

        # Use retrieved vertex indices to gather vertex signals required for interpolation
        mesh_signal = tf.reshape(
            tf.gather_nd(mesh_signal, vertex_indices, batch_dims=1),
            (
                -1,
                tf.shape(barycentric_coordinates)[1],
                self._template_size[0],
                self._template_size[1],
                3,
                self._feature_dim,
            ),
        )
        # (batch_shapes, vertices, n_radial, n_angular, input_dim)
        return tf.math.reduce_sum(
            tf.expand_dims(barycentric_coordinates[:, :, :, :, :, 1], axis=-1) * mesh_signal,
            axis=-2,
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
