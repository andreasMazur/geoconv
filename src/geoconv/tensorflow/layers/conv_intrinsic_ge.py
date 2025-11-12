from geoconv.tensorflow.layers.conv_base import ConvBase

import tensorflow as tf
import numpy as np


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


@tf.function
def into_polar_form(A):
    """Translates complex values into their polar form.

    Parameters
    ----------
    A: tf:Tensor
        A tensor that contains stacked complex values in its innermost dimension.

    Returns
    -------
    (tf.Tensor, tf.Tensor):
        Two tensors: the first contains the amplitudes and the second the phases.
    """
    A_shape = tf.shape(A)
    amplitudes = tf.linalg.norm(
        tf.reshape(A, (A_shape[0], A_shape[1], A_shape[2], A_shape[3], -1, 2)), ord=2, axis=-1
    )
    phases = tf.math.atan2(A[..., 1::2], A[..., ::2])
    return amplitudes, phases


class ConvHarmonicSurface(ConvBase):
    def __init__(self, output_dim, activation, template_radius, *args, **kwargs):
        super().__init__(
            template_radius=template_radius,
            include_kernel=False,
            activation=activation,
            *args,
            **kwargs
        )
        self.output_dim = output_dim
        self.n_complex_numbers = output_dim // 2
        self.all_angular_coordinates = tf.cast(self.template_vertices[0, :, 1], tf.float32)[..., None]

        # Set in build
        self.feature_dim = None
        self._amplitude_weights = None
        self._phase_m = None
        self._phase_b = None

    def build(self, inputs):
        signals_shape, bc_shape, angles_shape, _ = inputs
        super().build([signals_shape, bc_shape])

        # Initialize amplitude weights
        self._amplitude_weights = self.add_weight(
            name="radial_weights",
            shape=(self.n_complex_numbers, int(self.feature_dim / 2), self.n_radial),
            trainable=True,
        )

        # Initialize phase weights
        self._phase_m = self.add_weight(
            name="phase_m",
            shape=(self.n_complex_numbers, 1, 1, 1),
            trainable=True,
        )
        self._phase_b = self.add_weight(
            name="phase_b",
            shape=(self.n_complex_numbers, 1, 1, 1),
            trainable=True,
        )

    @tf.function
    def call(self, inputs):
        signals, bc, angles, input_rotation_orders = inputs

        # Gather the signals
        signals, bc_values = self._gather_signals(bc, signals)

        # Gather the rotations
        rotations = self._prepare_rotations(bc, angles, input_rotation_orders)

        # Apply rotations
        signals = complex_multiplication(rotations, signals)

        # Use patch operator
        signals = self._patch_operator(signals, bc_values)

        ### Determine polar form ###
        # signals    : (batch_shapes, vertices, radial, angular, input_dim)
        # amplitudes : (batch_shapes, vertices, radial, angular, input_dim / 2)
        # phases     : (batch_shapes, vertices, radial, angular, input_dim / 2)
        amplitudes, phases = into_polar_form(signals)

        ### Compute amplitude products ###
        # _amplitude_weights : (output_dim, input_dim / 2, radial)
        # amplitudes         : (batch_shapes, vertices, radial, angular, input_dim / 2)
        # results            : (batch_shapes, vertices, output_dim, radial, angular, input_dim / 2)
        amplitudes = tf.einsum("qfr,bkraf->bkqraf", self._amplitude_weights, amplitudes)

        # Apply activation on amplitudes ("magnitude non-linearity")
        amplitudes = self.activation_fn(amplitudes)

        ### Compute phase sums ###
        # _phase_m          : (output_dim, 1, 1, 1)
        # neighbor_angles   : (angular, 1)
        # _phase_b          : (output_dim, 1, 1, 1)
        # phases[..., None] : (batch_shapes, vertices,          1, radial, angular, input_dim / 2)
        # result            : (batch_shapes, vertices, output_dim, radial, angular, input_dim / 2)
        phases = self._phase_m * self.all_angular_coordinates + self._phase_b + phases[:, :, None, ...]

        ### Compute real and imaginary parts ###
        real = tf.reduce_sum(amplitudes * tf.math.cos(phases), axis=[-3, -2, -1])
        imaginary = tf.reduce_sum(amplitudes * tf.math.sin(phases), axis=[-3, -2, -1])

        # Return new complex numbers
        bc_shape = tf.shape(bc)
        return tf.reshape(
            tf.stack([real, imaginary], axis=-1), (bc_shape[0], bc_shape[1], self.output_dim)
        )

    @tf.function
    def _prepare_rotations(self, barycentric_coordinates, angles, input_rotation_orders):
        ### Gather the rotations angles ###
        # angles     : (n_batch, n_vertices, n_vertices)
        # bc[..., 1] : (n_batch, n_vertices, n_radial, n_angular, 3)
        # result     : (n_batch, n_vertices, n_radial, n_angular, 3)
        angles = tf.gather(angles, tf.cast(barycentric_coordinates[..., 1], dtype=tf.int32), batch_dims=2)

        ### Include rotation order ###
        # input_rotation_orders : (n_batch,          1,        1,         1, 1, input_dim / 2,)
        # angles[..., None]     : (n_batch, n_vertices, n_radial, n_angular, 3, 1)
        # result                : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        angles = input_rotation_orders[:, None, None, None, None, :] * angles[..., None]

        ### Calculate real parts ###
        real = tf.math.cos(angles)
        imaginary = tf.math.sin(angles)

        # real      : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        # imaginary : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        # result    : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        bc_shape = tf.shape(barycentric_coordinates)
        return tf.reshape(
            tf.stack([real, imaginary], axis=-1),
            (bc_shape[0], bc_shape[1], bc_shape[2], bc_shape[3], bc_shape[4], self.feature_dim)
        )

    def define_kernel_values(self, template_matrix):
        return None
