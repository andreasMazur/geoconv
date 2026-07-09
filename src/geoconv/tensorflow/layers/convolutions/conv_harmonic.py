from geoconv.tensorflow.layers.convolutions.conv_base import ConvBase

import tensorflow as tf


class ConvHarmonic(ConvBase):
    """This class implements the harmonic surface convolution.

    Original paper:
    > CNNs on surfaces using rotation-equivariant features
    > Ruben Wiersma and Elmar Eisemann and Klaus Hildebrandt
    > DOI: 10.1145/3386569.3392437
    """
    def __init__(self, output_dim, rotation_order, activation="linear", *args, **kwargs):
        super().__init__(
            include_kernel=False,
            activation=activation,
            *args,
            **kwargs
        )
        assert self.feature_dim % 2 == 0, "This layer requires the dimension of input features to be divisible by two."
        assert output_dim % 2 == 0, "This layer requires the dimension of output features to be divisible by two."

        self.output_dim = output_dim
        self.n_complex_num_output = output_dim // 2
        self.rotation_order = rotation_order

        # Set in build
        self._radial_weights = None
        self._radial_weights_center = None
        self._phase_offset = None
        self._all_angular_coordinates = None
        self.n_complex_num_input = None
        self.rotation_order_vector = None

    def build(self, inputs):
        signal_shape, bc_shape = inputs

        # Call build of parent class
        super().build([signal_shape, bc_shape])

        # Remember amount of complex input numbers
        self.n_complex_num_input = self.feature_dim // 2
        self.rotation_order_vector = tf.cast(
            tf.fill((self.n_complex_num_input,), self.rotation_order), tf.float32
        )

        # Initialize radial weights
        self._radial_weights = self.add_weight(
            name="radial_weights",
            shape=(self.n_radial, self.n_complex_num_output, self.n_complex_num_input),
            trainable=True
        )
        self._radial_weights_center = self.add_weight(
            name="radial_weights_center",
            shape=(self.n_complex_num_output, self.n_complex_num_input),
            trainable=True
        )

        # Initialize values required for computing the phase weight tensor
        self._phase_offset = self.add_weight(
            name="phase_offset",
            shape=(self.n_complex_num_output,),
            trainable=True
        )
        self._all_angular_coordinates = tf.cast(self.template_vertices[0, :, 1], tf.float32)

    @tf.function
    def create_phase_weight_tensor(self):
        """Creates phase weight tensor Phi.

        Returns
        -------
        tf.Tensor:
            The phase weight matrix computed with the current weights.
            Shape: (n_angular, output_dim, 2, 2)
        """
        # Compute arguments (i.e., angles) for trigonometric functions
        # _all_angular_coordinates : (n_angular,)
        # _phase_offset            : (output_dim / 2,)
        # angles                   : (n_angular, output_dim / 2)
        angles = (self.rotation_order * self._all_angular_coordinates)[:, None] + self._phase_offset[None, :]

        # Apply trigonometric functions
        # cos_matrix: (n_angular, output_dim / 2)
        # sin_matrix: (n_angular, output_dim / 2)
        cos_matrix = tf.cos(angles)
        sin_matrix = tf.sin(angles)

        # Create final weight matrix tensor of shape (n_angular, output_dim / 2, 2, 2)
        I = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        K = tf.constant([[0., -1.], [1., 0.]])
        return I * cos_matrix[..., None, None] + K * sin_matrix[..., None, None]

    @tf.function
    def create_phase_weight_tensor_center(self):
        """Creates phase weight tensor Phi for center vertices. I.e., theta = 0.

        Returns
        -------
        tf.Tensor:
            The phase weight matrix computed with the current weights.
            Shape: (output_dim / 2, 2, 2)
        """
        # Apply trigonometric functions on phase offset
        # cos_matrix: (output_dim / 2,)
        # sin_matrix: (output_dim / 2,)
        cos_matrix = tf.cos(self._phase_offset)
        sin_matrix = tf.sin(self._phase_offset)

        # Create final weight matrix tensor of shape (output_dim / 2, 2, 2)
        I = tf.constant([[1., 0.], [0., 1.]])
        K = tf.constant([[0., -1.], [1., 0.]])
        return I * cos_matrix[..., None, None] + K * sin_matrix[..., None, None]

    @tf.function
    def call(self, inputs):
        """Computes the harmonic surface convolution.

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            The first tensor has shape [n_batch, n_vertices, input_dim] and contains the signals for each vertex. The
            second tensor has shape [n_batch, n_vertices, n_radial, n_angular, 3, 2] and contains the barycentric
            coordinates.

        Returns
        -------
        tf:Tensor
            A tensor of size [n_batch, n_vertices, output_dim], containing the new signal-embeddings for each mesh
            vertex.
        """
        # signals : (n_batch, n_vertices, input_dim)
        # bc      : (n_batch, n_vertices, n_radial, n_angular, 3, 3)
        signals, bc = inputs

        # Get transported and interpolated feature vectors at each template vertex
        # neighbor_signals : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        neighbor_signals = self._signal_pullback_with_parallel_transport(signals, bc, self.rotation_order_vector)

        # Get phase weight tensor
        # phase_weights: (n_angular, output_dim / 2, 2, 2)
        phase_weights_neigh = self.create_phase_weight_tensor()

        # Aggregate neighbors
        # radial_weights      : (n_radial, output_dim / 2, input_dim / 2)
        # phase_weights_neigh : (n_angular, output_dim / 2, 2, 2)
        # neighbor_signals    : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        # conv_neigh          : (n_batch, n_vertices, output_dim / 2, 2)
        conv_neigh = tf.einsum(
            "rqf,aqxy,bkrafy->bkqx", self._radial_weights, phase_weights_neigh, neighbor_signals
        )

        # Reshape vertex signals into their geometric components
        # signals : (n_batch, n_vertices, input_dim / 2, 2)
        signals_shape = tf.shape(signals)
        signals = tf.reshape(signals, (signals_shape[0], signals_shape[1], self.n_complex_num_input, 2))

        # Compute self connections
        # radial_weights_center : (output_dim / 2, input_dim / 2)
        # phase_weights_center  : (output_dim / 2, 2, 2)
        # signals               : (n_batch, n_vertices, input_dim / 2, 2)
        # conv_center           : (n_batch, n_vertices, output_dim / 2, 2)
        phase_weights_center = self.create_phase_weight_tensor_center()
        conv_center = tf.einsum(
            "qf,qxy,bkfy->bkqx", self._radial_weights_center, phase_weights_center, signals
        )

        # Add self-connection contributions to neighbor aggregation for complete conv result
        # TODO: Consider removing normalization by amount of template vertices
        # conv_center : (n_batch, n_vertices, output_dim / 2, 2)
        # conv_neigh  : (n_batch, n_vertices, output_dim / 2, 2)
        # result      : (n_batch, n_vertices, output_dim / 2, 2)
        result = (conv_center + conv_neigh) / tf.cast(1 + self.n_radial + self.n_angular, tf.float32)

        # Apply magnitude activation
        # result_amp : (n_batch, n_vertices, output_dim / 2)
        # result     : (n_batch, n_vertices, output_dim / 2, 2)
        result_amp = tf.maximum(tf.linalg.norm(result, axis=-1), 1e-6)
        result = self.activation_fn(result_amp)[..., None] * tf.math.divide_no_nan(result, result_amp[..., None])

        # Return output in original shape
        # (n_batch, n_vertices, output_dim)
        return tf.reshape(result, (signals_shape[0], signals_shape[1], self.output_dim))

    def define_kernel_values(self, template_matrix):
        return None

    def get_config(self):
        """Adds class relevant to the config-dictionary of the base 'Layer' class.

        Returns
        -------
        dict:
            The class configuration in the form of a dictionary.
        """
        base_config = super().get_config()

        # Update parent class dict
        class_config = {
            "output_dim": self.output_dim,
            "rotation_order": self.rotation_order
        }
        base_config.update(class_config)

        # Prevent double keyword argument (init sets 'include_kernel' to false)
        del base_config["include_kernel"]
        return base_config
