from geoconv.tensorflow.layers.conv_base import ConvBase

import tensorflow as tf


@tf.function
def compute_amplitude_and_normalize(A):
    """Computes the amplitude complex numbers stored in a tensor and normalizes those.

    This function assumes that the innermost dimension contains stacked 2-dimensional vectors, each of which contains
    one entry for the real- and one entry for the imaginary part of a complex number. E.g:

    A = [
        [1, 2, 3, 4], => Imaginary numbers: 1 + 2i AND 3 + 4i
        [3, 4, 1, 2], => Imaginary numbers: 3 + 4i AND 1 + 2i
        [5, 6, 10, 11], => Imaginary numbers: 5 + 6i AND 10 + 11i
    ] whereby 'A' with 'shape(A) = (a = 3, b = 4)' thus contains a * (b / 2) complex numbers.

    Parameters
    ----------
    A: tf.Tensor
        The tensor that contains stacked complex numbers in its last dimension.

    Returns
    -------
    (tf.Tensor, tf:tensor):
        Two tensors: (i) a tensor containing complex numbers with amplitude 1 in its last two dimensions and (ii) a
        tensor containing the original amplitudes of the complex numbers in its last dimension.
    """
    A_shape = tf.shape(A)
    A = tf.reshape(A, tf.concat([A_shape[:-1], [-1], [2]], axis=-1))
    A_amplitudes = tf.linalg.norm(A, ord=2, axis=-1)
    A = tf.math.divide_no_nan(A, A_amplitudes[..., None])
    return A, A_amplitudes


@tf.function
def complex_multiplication(A, B):
    """Computes the complex multiplication between the innermost dimension of two tensors.

    This function assumes that the innermost dimension contains stacked 2-dimensional vectors, each of which contains
    one entry for the real- and one entry for the imaginary part of a complex number. E.g:

    A = [
        [1, 2, 3, 4], => Imaginary numbers: 1 + 2i AND 3 + 4i
        [3, 4, 1, 2], => Imaginary numbers: 3 + 4i AND 1 + 2i
        [5, 6, 10, 11], => Imaginary numbers: 5 + 6i AND 10 + 11i
    ] whereby 'A' with 'shape(A) = (a = 3, b = 4)' thus contains a * (b / 2) complex numbers.

    Parameters
    ----------
    A: tf.Tensor
        The first tensor for the complex multiplication. Has to have the same shape as 'B'.
    B: tf.Tensor
        The second tensor for the complex multiplication. Has to have the same shape as 'A'.

    Returns
    -------
    tf.Tensor:
        A tensor of same shape as 'A' and 'B' that contains the complex product of 'A' and 'B'.
    """
    assert A.shape == B.shape, "A and B need to have the same dimension."
    A_shape = tf.shape(A)

    A_real = A[..., ::2]
    A_complex = A[..., 1::2]
    B_real = B[..., ::2]
    B_complex = B[..., 1::2]

    C_real = A_real * B_real - A_complex * B_complex
    C_complex = A_real * B_complex + A_complex * B_real
    return tf.reshape(tf.stack([C_real, C_complex], axis=-1), A_shape)


class ConvGaugeEquiv(ConvBase):
    def __init__(self, output_dim, activation, template_radius, *args, **kwargs):
        super().__init__(
            template_radius=template_radius,
            include_kernel=False,
            activation=activation,
            *args,
            **kwargs
        )
        assert output_dim > 0 and output_dim % 2 == 0, "The output dimensionality has to be even!"
        self.output_dim = output_dim
        self.n_complex_numbers = output_dim // 2

        # Set in build
        self.feature_dim = None
        self.all_angular_coordinates = None
        self._amplitude_weights_center = None
        self._phase_b_center = None
        self._amplitude_weights = None
        self._phase_m = None
        self._phase_b = None

    def build(self, inputs):
        signals_shape, bc_shape, rotations_shape, orders_shape = inputs
        super().build([signals_shape, bc_shape])
        assert self.feature_dim > 0 and self.feature_dim % 2 == 0, \
            f"The input dimensionality ({self.feature_dim}) has to be even!"

        # Require template vertices from super().build()
        self.all_angular_coordinates = tf.cast(self.template_vertices[0, :, 1], tf.float32)

        ### Center weights ###
        self._amplitude_weights_center = self.add_weight(
            name="center_radial_weights",
            shape=(self.n_complex_numbers, int(self.feature_dim / 2)),
            trainable=True,
        )
        self._phase_b_center = self.add_weight(
            name="phase_b",
            shape=(self.n_complex_numbers,),
            trainable=True
        )

        ### Neighbor weights ###
        # Initialize amplitude weights
        self._amplitude_weights = self.add_weight(
            name="radial_weights",
            shape=(self.n_complex_numbers, self.n_radial, int(self.feature_dim / 2)),
            trainable=True,
        )

        # Initialize phase weights
        self._phase_m = self.add_weight(
            name="phase_m",
            shape=(self.n_complex_numbers,),
            trainable=True
        )
        self._phase_b = self.add_weight(
            name="phase_b",
            shape=(self.n_complex_numbers, 1),
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        # signals               : (batch, n_vertices, input_dim)
        # bc                    : (batch, n_vertices, n_radial, n_angular, 3, 2)
        # angles                : (batch, n_vertices, n_vertices)
        # input_rotation_orders : (batch, input_dim / 2)
        signals, bc, angles, input_rotation_orders = inputs

        # center_result : (batch, n_vertices, output_dim / 2, 2)
        center_result = self.conv_center(signals)

        # neighbor_result : (batch, n_vertices, output_dim / 2, 2)
        neighbor_result = self.conv_neighbor(signals, bc, angles, input_rotation_orders)

        ### Add results ###
        result = center_result + neighbor_result

        ### Apply magnitude activation ###
        # result : (batch, n_vertices, output_dim)
        result_amp = tf.linalg.norm(result, axis=-1)
        result = self.activation_fn(result_amp)[..., None] * tf.math.divide_no_nan(result, result_amp[..., None])

        ### Reshape to original shape ###
        # result : (batch, n_vertices, output_dim)
        result_shape = tf.shape(result)
        result = tf.reshape(result, (result_shape[0], result_shape[1], self.output_dim))

        return result, tf.tile(self._phase_m[None, ...], multiples=[result_shape[0], 1])

    @tf.function
    def conv_center(self, signals):
        # Create phase weight tensor (bias only, assume 'theta = 0' in origin
        P_w_center = self.create_phase_weight_tensor_center()

        # Get normalized signals and their amplitudes
        signals, signal_amplitudes = compute_amplitude_and_normalize(signals)
        center_conv = tf.einsum(
            "qf,bkf,qij,bkfj->bkqi", self._amplitude_weights_center, signal_amplitudes, P_w_center, signals
        )
        return center_conv

    @tf.function
    def conv_neighbor(self, signals, bc, angles, input_rotation_orders):
        # Gather the signals
        # signals   : (batch, n_vertices, n_radial, n_angular, 3, input_dim)
        # bc_values : (batch, n_vertices, n_radial, n_angular, 3)
        signals, bc_values = self._gather_signals(bc, signals)

        # Gather the rotations
        # rotations : (batch, n_vertices, n_radial, n_angular, 3, 2)
        rotations = self._prepare_rotations(bc, angles, input_rotation_orders)

        # Apply rotations
        # signals : (batch, n_vertices, n_radial, n_angular, 3, input_dim)
        signals = complex_multiplication(rotations, signals)

        # Use patch operator
        # signals : (batch, n_vertices, n_radial, n_angular, input_dim)
        signals = self._patch_operator(signals, bc_values)

        # Compute amplitudes
        # signals          : (batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        # signal_amplitudes: (batch, n_vertices, n_radial, n_angular, input_dim / 2)
        signals, signal_amplitudes = compute_amplitude_and_normalize(signals)

        # Create phase weight tensor
        P_w = self.create_phase_weight_tensor()

        # Compute convolution
        neighbor_conv = tf.einsum(
            "qrf,bkraf,qaij,bkrafj->bkqi", self._amplitude_weights, signal_amplitudes, P_w, signals
        )
        return neighbor_conv

    @tf.function
    def create_phase_weight_tensor_center(self):
        cos_matrix = tf.cos(self._phase_b_center)
        sin_matrix = tf.sin(self._phase_b_center)
        I = tf.constant([[1., 0.], [0., 1.]])
        K = tf.constant([[0., -1.], [1., 0.]])
        return I * cos_matrix[..., None, None] + K * sin_matrix[..., None, None]

    @tf.function
    def create_phase_weight_tensor(self):
        weight_coord_matrix = tf.einsum("i,j->ij", self._phase_m, self.all_angular_coordinates) + self._phase_b
        cos_matrix = tf.cos(weight_coord_matrix)
        sin_matrix = tf.sin(weight_coord_matrix)
        I = tf.constant([[1., 0.], [0., 1.]])
        K = tf.constant([[0., -1.], [1., 0.]])
        return I * cos_matrix[..., None, None] + K * sin_matrix[..., None, None]

    @tf.function
    def _prepare_rotations(self, barycentric_coordinates, angles, input_rotation_orders):
        ### Gather the rotations angles ###
        # angles     : (n_batch, n_vertices, n_vertices)
        # bc[..., 1] : (n_batch, n_vertices, n_radial, n_angular, 3)
        # result     : (n_batch, n_vertices, n_radial, n_angular, 3)
        angles = tf.gather(angles, tf.cast(barycentric_coordinates[..., 1], dtype=tf.int32), batch_dims=2)

        ### Include rotation order ###
        # input_rotation_orders : (n_batch,          1,        1,         1, 1, input_dim / 2)
        # angles[..., None]     : (n_batch, n_vertices, n_radial, n_angular, 3, 1)
        # result                : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        angles = input_rotation_orders[:, None, None, None, None, :] * angles[..., None]

        ### Calculate real and imaginary parts ###
        real = tf.math.cos(angles)
        imaginary = tf.math.sin(angles)

        # real      : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        # imaginary : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        # result    : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2)
        complex_values = tf.stack([real, imaginary], axis=-1)

        # complex_values : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2)
        # result         : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        bc_shape = tf.shape(barycentric_coordinates)
        return tf.reshape(
            complex_values,
            (bc_shape[0], bc_shape[1], bc_shape[2], bc_shape[3], bc_shape[4], self.feature_dim)
        )

    def define_kernel_values(self, template_matrix):
        return None
