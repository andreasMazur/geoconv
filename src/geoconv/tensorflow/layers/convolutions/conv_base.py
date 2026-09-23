from geoconv.preprocessing.bc.bc_utils import create_template_matrix

from abc import abstractmethod

import tensorflow as tf
import numpy as np


class ConvBase(tf.keras.layers.Layer):
    """The base layer for any surface convolution layer in GeoConv.

    Attributes
    ----------
    template_radius: float
        The geodesic radius of the discretized template.
    n_radial: int
        The amount of radial coordinates for the discretized template.
    n_angular: int
        The amount of angular coordinates for the discretized template.
    template_vertices: torch.Tensor
        The template vertices of the discretized template.
    feature_dim: int
        The input feature dimension.
    activation_fn: callable
        The activation function.
    include_kernel: bool
        Whether to include a kernel/weighting function that interpolates the signals gathered at the template vertices.
    kernel: tf.Tensor | None
        The kernel/weight function that interpolates the signals at the template vertices.
    """
    def __init__(self, template_radius, include_kernel, activation, *args, **kwargs,):
        """Initializes the object.

        Parameters
        ----------
        template_radius: float
            The template radius.
        include_kernel: bool
             Whether to include a kernel/weighting function that interpolates the signals gathered at the template
             vertices.
        activation: str
            The activation function's name.
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        super().__init__(*args, **kwargs)
        # Configure template
        self.template_radius = template_radius

        # Configure activation function
        self.activation = activation
        self.activation_fn = tf.keras.layers.Activation(self.activation)

        # Configure kernel
        self.include_kernel = include_kernel

        # Set in build the moment inputs have been seen
        self.feature_dim = None
        self.n_radial = None
        self.n_angular = None
        self.template_vertices = None
        self.kernel = None

    def build(self, inputs):
        """Builds the layer weights.
            
        Parameters
        ----------
        inputs: tuple
            Two tensor shapes for both the mesh signal and barycentric coordinates.
        """
        signal_shape, barycentric_coordinates_shape = inputs
        self.feature_dim = signal_shape[-1]
        self.n_radial = barycentric_coordinates_shape[-4]
        self.n_angular = barycentric_coordinates_shape[-3]
        self.template_vertices = tf.constant(
            create_template_matrix(
                self.n_radial,
                self.n_angular,
                radius=self.template_radius,
                in_cart=False
            )
        )
        if self.include_kernel:
            self.kernel = tf.cast(
                self.define_kernel_values(self.template_vertices.numpy()), tf.float32
            )

    @tf.function
    def _gather_signals(self, barycentric_coordinates, mesh_signal, bc_with_angles=False):
        """Gathers required feature vectors and associates those to given barycentric coordinates.

        Parameters
        ----------
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates tensor.
        mesh_signal: tf.Tensor
            The feature vectors at the mesh vertices.
        bc_with_angles: bool
            Whether the barycentric coordinates have angles concatenated to them.

        Returns
        -------
        (tf.Tensor, tf.Tensor):
            A tensor of shape (n_batch, n_vertices, n_radial, n_angular, 3, input_dim) that contains the required
            feature vectors and their according interpolation values in a tensor of shape
            (n_batch, n_vertices, n_radial, n_angular, 3).
        """
        # n_batch, n_vertices, n_radial, n_angular, 3, 2
        bc_shape = tf.shape(barycentric_coordinates)

        # (n_batch, n_vertices * n_radial * n_angular * 3)
        if bc_with_angles:
            bc_values, bc_indices, _ = tf.unstack(barycentric_coordinates, axis=-1)
        else:
            bc_values, bc_indices = tf.unstack(barycentric_coordinates, axis=-1)
        bc_indices = tf.cast(tf.reshape(bc_indices, (bc_shape[0], -1)), tf.int32)

        # (n_batch, n_vertices * n_radial * n_angular * 3, input_dim)
        mesh_signal = tf.gather(mesh_signal, bc_indices, batch_dims=1)

        # (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        mesh_signal = tf.reshape(
            mesh_signal, (bc_shape[0], bc_shape[1], bc_shape[2], bc_shape[3], 3, self.feature_dim)
        )
        return mesh_signal, bc_values

    @tf.function
    def _patch_operator(self, mesh_signal, barycentric_coordinates):
        """Implementation of the patch-operator: weighting and signal-pullback.

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The feature vectors that shall be interpolated at the template vertices.
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices.

        Returns
        -------
        tf.Tensor:
            A tensor containing weighted (in case pre-defined kernels are used) and interpolated mesh signals of shape
            (batch_shapes, vertices, radial, angular, input_dim).
        """
        # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
        interpolations = self._signal_pullback(mesh_signal, barycentric_coordinates)

        if self.include_kernel:
            # Weight matrix  : (radial, angular, radial, angular)
            # interpolations : (batch_shapes, vertices, radial, angular, input_dim)
            # Result         : (batch_shapes, vertices, radial, angular, input_dim)
            return tf.einsum("raxy,skxyf->skraf", self.kernel, interpolations)
        else:
            return interpolations

    @tf.function
    def _signal_pullback(self, mesh_signal, barycentric_coordinates):
        """Implementation of signal-pullback: Weighting of gathered and interpolated mesh signals.

        Parameters
        ----------
        mesh_signal: tf.Tensor
            The signal values at the template vertices.
        barycentric_coordinates: tf.Tensor
            The barycentric coordinates for the template vertices.

        Returns
        -------
        tf.Tensor:
            A tensor containing interpolated feature vectors at the template vertices of shape
            (n_batch, n_vertices, n_radial, n_angular, input_dim).
        """

        # (n_batch, n_vertices, n_radial, n_angular, input_dim)
        return tf.reduce_sum(tf.expand_dims(barycentric_coordinates, axis=-1) * mesh_signal, axis=-2)

    @tf.function
    def _signal_pullback_with_parallel_transport(self, signals, bc, rotation_order_vector):
        """Wrapper function for feature gathering, parallel transport and interpolation.

        Parameters
        ----------
        signals: tf.Tensor
            The surface signal.
            Shape: (batch, n_vertices, input_dim)
        bc: tf.Tensor
            The barycentric coordinates tensor with angles.
            Shape: (batch, n_vertices, n_radial, n_angular, 3, 3)
        rotation_order_vector: tf.Tensor
            A vector describing the rotation orders of the individual geometric components.
            Shape: (input_dim / 2)

        Returns
        -------
        tf.Tensor:
            Transported and interpolated feature vectors for each template vertex split into individual geometric
            components.
            Shape: (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        """
        # Get template vertex interpolations
        # neighbor_signals : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim)
        # bc_coefficients  : (n_batch, n_vertices, n_radial, n_angular, 3)
        neighbor_signals, bc_coefficients = self._gather_signals(bc, signals, bc_with_angles=True)

        # Reshape gathered signals into their geometric components
        # neighbor_signals : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2)
        signals_shape = tf.shape(neighbor_signals)
        n_geometric_components = tf.shape(rotation_order_vector)[0]
        neighbor_signals = tf.reshape(
            neighbor_signals,
            (
                signals_shape[0],
                signals_shape[1],
                self.n_radial,
                self.n_angular,
                3,
                n_geometric_components,
                2,
            ),
        )

        # Prepare rotation matrices for parallel transport
        # rotation_matrices : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        rotation_matrices = self.create_rotation_matrices(bc, rotation_order_vector)

        # Transport via rotation and interpolate signals at template vertices
        # bc_coefficients   : (n_batch, n_vertices, n_radial, n_angular, 3)
        # rotation_matrices : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        # neighbor_signals  : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2)
        # result            : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        return tf.einsum(
            "bkral,bkralnxy,bkralny->bkranx", bc_coefficients, rotation_matrices, neighbor_signals
        )

    @tf.function
    def create_rotation_matrices(self, bc, rotation_order):
        """Creates rotation matrices from given angles for parallel transport.

        Rotation matrix used:
            [cos ng, -sin ng]
            [sin ng, cos ng],
        whereby 'n = rotation_order'. If 'n = 0' rotation matrix becomes unit matrix
        and leaves features invariant.

        Parameters
        ----------
        bc: tf.Tensor
            The barycentric coordinates tensor.
        rotation_order: tf.Tensor
            The rotation order vector contains coefficients for how fast individual components of a
            feature vector rotate. It has shape (input_dim / 2).

        Returns
        -------
        tf.Tensor:
            Rotation matrices for the parallel transport.
            Shape: (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        """
        ### Gather the rotations angles ###
        # The parallel transport of a feature vector corresponds to a rotation of -\theta, when \theta
        # is the rotation of the x-axis in coordinate frame 1 to the x-axis of coordinate frame 2.
        # result     : (n_batch, n_vertices, n_radial, n_angular, 3)
        angles = -bc[..., -1]

        ### Include rotation order ###
        # rotation_order : (      1,          1,        1,         1, 1, input_dim / 2)
        # angles         : (n_batch, n_vertices, n_radial, n_angular, 3,             1)
        # result         : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        angles = rotation_order[None, None, None, None, None, :] * angles[..., None]

        ### Translate angles to rotation matrices ###
        # angles : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2)
        # result : (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        C = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        S = tf.constant([[0., -1.], [1., 0.]])
        rot_matrices = tf.cos(angles)[..., None, None] * C + tf.sin(angles)[..., None, None] * S

        # 'rot_matrices': (n_batch, n_vertices, n_radial, n_angular, 3, input_dim / 2, 2, 2)
        return rot_matrices

    def get_config(self):
        """Adds class relevant to the config-dictionary of the base 'Layer' class.

        Returns
        -------
        dict:
            The class configuration in the form of a dictionary.
        """
        base_config = super().get_config()
        class_config = {
            "template_radius": self.template_radius,
            "include_kernel": self.include_kernel,
            "activation": self.activation,
        }
        base_config.update(class_config)
        return base_config

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
