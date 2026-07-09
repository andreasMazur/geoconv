from geoconv.tensorflow.layers.convolutions.conv_base import ConvBase
from geoconv.utils.sine_cosine_locs import get_sin_and_cosine_locs_neigh, get_sin_and_cosine_locs_self

import tensorflow as tf
import numpy as np


@tf.function
def construct_angle_tensor(gamma_in, gamma_out, angles, phase_weights):
    """Constructs the angle tensors for the basis kernels.

    Parameters
    ----------
    gamma_in: tf.Tensor
        A tensor of shape (d_in,) which contains the output types.
    gamma_out: tf.Tensor
        A tensor of shape (d_out,) which contains the input types.
    angles: tf.Tensor
        A tensor of shape (n_angular,) which contains all angles that shall be considered.
    phase_weights: tf.Tensor
        A tensor of shape (d_out,) which contains the phase weights.

    Returns
    -------
    tf.Tensor:
        A tensor of shape (d_out, d_in, 4, n_angular) that contains the angle inputs for the irreps tensors.
    """
    ### Create all input-output-pairs  ###
    # 'gamma_out_tiled': (d_out, d_in)
    d_in = tf.shape(gamma_in)[0]
    gamma_out_tiled = tf.tile(gamma_out[:, None], (1, d_in))

    # 'gamma_in_tiled': (d_out, d_in)
    d_out = tf.shape(gamma_out)[0]
    gamma_in_tiled = tf.tile(gamma_in[None, :], (d_out, 1))

    # 'gamma_out_in': (d_out, d_in, 2)
    gamma_out_in = tf.stack([gamma_out_tiled, gamma_in_tiled], axis=-1)

    ### Compute rotation orders (pay attention to +-) ###
    # 'gamma': (d_in * d_out, 5)
    gamma_out_in = tf.reshape(gamma_out_in, (d_out * d_in, 2))
    gamma_out = gamma_out_in[:, 0]
    gamma_in = gamma_out_in[:, 1]

    zeros = tf.zeros_like(gamma_in)
    ones = tf.ones_like(gamma_in)

    mask1 = gamma_out == 0
    mask2 = (gamma_out != 0) & (gamma_in == 0)

    case1 = tf.stack(
        [
            gamma_in,
            gamma_in,
            zeros,
            zeros,
            tf.cast(gamma_in > 0, tf.float32)  # Use phase weight if gamma > 0
        ], axis=-1
    )
    case2 = tf.stack(
        [
            gamma_out,
            gamma_out,
            zeros,
            zeros,
            ones  # Since gamma' > 0, use phase weight
        ], axis=-1
    )
    case3 = tf.stack(
        [
            gamma_out - gamma_in,
            gamma_out - gamma_in,
            gamma_out + gamma_in,
            gamma_out + gamma_in,
            ones  # Since gamma' > 0, use phase weight
        ], axis=-1
    )
    gamma = tf.where(
        mask1[:, None],
        case1,  # \rho_[0|gamma] -> \rho_0
        tf.where(
            mask2[:, None],
            case2,  # \rho_0 -> \rho_gamma'
            case3   # \rho_gamma -> \rho_gamma'
        )
    )

    # 'gamma': (d_out, d_in, 5)
    gamma = tf.reshape(gamma, [d_out, d_in, 5])

    ### Only in cases where gamma or gamma' are non-zero use phase weights ###
    # 'mask': (d_out, d_in, 1)
    mask = gamma[..., -1][..., None]

    # 'gamma': (d_out, d_in, 4)
    gamma = gamma[..., :-1]

    # Return angles: (d_out, d_in, 4, n_angular)
    return gamma[..., None] * angles[None, None, None, :] + phase_weights[:, None, None, None] * mask[..., None]


@tf.function
def get_kernel_neigh(gamma_in, gamma_out, sine_and_cosine_locs, angles, phase_weights):
    """Constructs a tensor that contains all basis kernels for GEM-CNNs.

    Parameters
    ----------
    gamma_in: tf.Tensor
        A tensor of shape (d_in,) which contains the output types.
    gamma_out: tf.Tensor
        A tensor of shape (d_out,) which contains the input types.
    sine_and_cosine_locs: tf.Tensor
        A tensor of shape (2, d_out, d_in, 4, 2, 2) which contains the projection matrices for sines and cosines.
    angles: tf.Tensor
        A tensor of shape (n_angular,) which contains all angles that shall be considered.
    phase_weights: tf.Tensor
        A tensor of shape (d_out,) which contains the phase weights.

    Returns
    -------
    tf.Tensor:
        A tensor of shape (d_out, d_in, 4, n_angular, 2, 2) which contains all basis kernels required for the GEM-CNN
        convolution.
    """
    # 'angle_tensor': (d_out, d_in, 4, n_angular)
    angle_tensor = construct_angle_tensor(gamma_in, gamma_out, angles, phase_weights)
    cosines = tf.cos(angle_tensor)
    sines = tf.sin(angle_tensor)

    # return: (d_out, d_in, 4, n_angular, 2, 2)
    return (
            sines[..., None, None] * sine_and_cosine_locs[0, :, :, :, None, ...] + \
            cosines[..., None, None] * sine_and_cosine_locs[1, :, :, :, None, ...]
    )


@tf.function
def get_kernel_self(gamma_in, gamma_out, sine_and_cosine_locs, phase_weights):
    """Constructs a tensor that contains all self-connection basis kernels for GEM-CNNs.

    Parameters
    ----------
    gamma_in:
        A tensor of shape (d_in,) which contains the input types.
    gamma_out:
        A tensor of shape (d_out,) which contains the output types.
    sine_and_cosine_locs:
    phase_weights:

    Returns
    -------
    tf.Tensor:
        The basis kernels for kernel construction.
    """
    ### Create all input-output-pairs  ###
    # 'gamma_out_tiled': (d_out, d_in)
    d_in = tf.shape(gamma_in)[0]
    gamma_out_tiled = tf.tile(gamma_out[:, None], (1, d_in))
    # 'gamma_in_tiled': (d_out, d_in)
    d_out = tf.shape(gamma_out)[0]
    gamma_in_tiled = tf.tile(gamma_in[None, :], (d_out, 1))
    # 'gamma_out_in': (d_out, d_in, 2)
    gamma_out_in = tf.stack([gamma_out_tiled, gamma_in_tiled], axis=-1)

    # Determine angles via case distinction: IF gamma_out == gamma_in THEN phase_weight ELSE 0
    # 'gamma_out_in'  : (d_out, d_in, 2)
    # 'phase_weights' : (d_out, )
    # 'angles'        : (d_out, d_in)
    angles = tf.cast(tf.equal(gamma_out_in[..., 0], gamma_out_in[..., 1]), tf.float32) * phase_weights[:, None]

    # Create basis-kernels for linear combination
    # Note: sines/cosine > 0 might be set to zero by 'sine_and_cosine_locs' tensor
    # sine_and_cosine_locs : (2, d_out, d_in, 2, 2, 2)
    # 'angles'             : (d_out, d_in)
    # 'return'             : (d_out, d_in, 2, 2, 2)
    return (
        tf.sin(angles[..., None, None, None]) * sine_and_cosine_locs[0] + \
        tf.cos(angles[..., None, None, None]) * sine_and_cosine_locs[1]
    )


class ConvGEM(ConvBase):
    """This class implements the Gauge-equivariant Mesh convolution.

    Original paper:
    > Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs
    > Pim De Haan and Maurice Weiler and Taco Cohen and Max Welling
    > URL: https://openreview.net/forum?id=Jnspzp-oIZE
    """
    def __init__(self, input_types, output_types, activation="linear", *args, **kwargs):
        super().__init__(
            include_kernel=False,
            activation=activation,
            *args,
            **kwargs
        )

        # Check that input- and output vectors are divisible by 2
        assert self.feature_dim % 2 == 0, "This layer requires the dimension of input features to be divisible by two."

        self.input_types = tf.constant(input_types, dtype=tf.float32)
        self.input_dim_halve = self.input_types.shape[0]
        self.output_types = tf.constant(output_types, dtype=tf.float32)
        self.output_dim_halve = self.output_types.shape[0]

        # Set in build
        self.all_angular_coordinates = None
        self.K_neigh = None
        self.V_neigh = None
        self.K_self = None
        self.V_self = None

    def build(self, inputs):
        signal_shape, bc_shape = inputs

        # Call build of parent class
        super().build([signal_shape, bc_shape])

        # Remember all angular coordinates
        self.all_angular_coordinates = tf.cast(self.template_vertices[0, :, 1], tf.float32)

        ### In preparation for 'W_neigh' ###
        # 'K_neigh': (output_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        self.K_neigh = get_kernel_neigh(
            gamma_in=self.input_types,
            gamma_out=self.output_types,
            sine_and_cosine_locs=tf.constant(
                get_sin_and_cosine_locs_neigh(self.input_types.numpy(), self.output_types.numpy())
            ),
            angles=self.all_angular_coordinates,
            phase_weights=tf.fill(dims=self.output_dim_halve, value=0.)
        )
        # 'V_neigh': (output_dim / 2, input_dim / 2, 4)
        self.V_neigh = self.add_weight(
            name="V_neigh",
            shape=(self.output_dim_halve, self.input_dim_halve, 4),
            trainable=True
        )

        ### In preparation for 'W_self' ###
        # 'K_self': (output_dim / 2, input_dim / 2, 2, 2, 2)
        self.K_self = tf.constant(
            get_sin_and_cosine_locs_self(gamma_in=self.input_types.numpy(), gamma_out=self.output_types.numpy())
        )
        # 'V_self' : (output_dim / 2, input_dim / 2, 2)
        self.V_self = self.add_weight(
            name="V_self",
            shape=(self.output_dim_halve, self.input_dim_halve, 2)
        )

    @tf.function(jit_compile=True)
    def get_self_connection_embeddings(self, signals, V_self, K_self):
        """Constructs the kernel matrix from trainable weights and irreps, and subsequently embeds the input signal.

        Note:
        A complex feature consists of two scalars. Therefore, considering an input feature vector that has 'input_dim'
        many scalar entries, it only consists out of 'input_dim / 2' features.

        Parameters
        ----------
        signals: tf.Tensor
            The feature vectors at the convolution centers. It has shape (batch_dim, n_vertices, d_in, 2). The batch
            dimension is the amount of shapes. The number 'n_vertices' is the number of vertices. The last two
            dimensions 'd_in' and '2' describe a complex feature vector. Each feature consists of two scalars.
        V_self: tf.Tensor
            The trainable weights for the self connections. It has shape (d_out, d_in, 2). The first dimension
            refers to 'd_out' many complex output features that the layer shall compute. The second dimension refers to
            the amount of complex input features that the layer receives. This tensor stores 2 trainable weights per
            input-output pair for the linear combination of the basis kernels in its last dimensions.
        K_self: tf.Tensor
            The kernel matrix for self connections. It has shape (d_out, d_in, s=2, 2, 2). The first dimension
            refers to 'd_out' many complex output features that the layer shall compute. The second dimension refers to
            the amount of complex input features that the layer receives. The dimension 's' refers to two basis kernel
            matrices, that self connections use. The last two dimensions represent the dimensions of the 2x2 basis
            kernel matrices.

        Returns
        -------
        tf.Tensor:
            The embedded signal. It has shape (n_batch, n_vertices, d_out, 2). See 'signals'-argument for more detailed
            description about what each dimension refers to.
        """
        ### Determine weight matrix for neighbor aggregation ###
        # 'V_self' : (output_dim / 2, input_dim / 2, 2)
        # 'K_self' : (output_dim / 2, input_dim / 2, 2, 2, 2)
        # 'W_self' : (output_dim / 2, input_dim / 2, 2, 2)
        W_self = tf.einsum("mns,mnsxy->mnxy", V_self, K_self)

        ### Compute neighbor embeddings ###
        # 'W_self'  : (output_dim / 2, input_dim / 2, 2, 2)
        # 'signals' : (n_batch, n_vertices, input_dim / 2, 2)
        # return    : (n_batch, n_vertices, output_dim / 2, 2)
        return tf.einsum("mnxy,bkny->bkmx", W_self, signals)

    @tf.function(jit_compile=True)
    def get_neigh_embeddings(self, interpolations, V_neigh, K_neigh, neighbor_aggregation=True):
        """Constructs the kernel matrix from trainable weights and irreps, and subsequently embeds the input signal.

        Parameters
        ----------
        interpolations: tf.Tensor
            The interpolated feature vectors at the template vertices.
        V_neigh: tf.Tensor
            The trainable weights for the neighbor aggregation.
        K_neigh: tf.Tensor
            The kernel tensor for the neighbor aggregation.
        neighbor_aggregation: bool
            Whether to sum over the neighbor embeddings.

        Returns
        -------
        tf.Tensor:
            Embeddings computed using embedded neighbor features.
        """
        ### Determine weight matrix for neighbor aggregation ###
        # 'V_neigh' : (output_dim / 2, input_dim / 2, 4)
        # 'K_neigh' : (output_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        # 'W_self'  : (output_dim / 2, input_dim / 2   , n_angular, 2, 2)
        W_neigh = tf.einsum("mns,mnsaxy->mnaxy", V_neigh, K_neigh)

        ### Compute neighbor embeddings ###
        # W_neigh        : (output_dim / 2, input_dim / 2, n_angular, 2, 2)
        # interpolations : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        if neighbor_aggregation:
            # return : (n_batch, n_vertices, output_dim / 2, 2)
            return tf.einsum("mnaxy,bkrany->bkmx", W_neigh, interpolations)
        else:
            # return : (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
            return tf.einsum("mnaxy,bkrany->bkramx", W_neigh, interpolations)

    @tf.function
    def call(self, inputs):
        """Prepares format of input signal, computes interpolations and returns GEM-CNN results.

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            Two tensors. The first tensor contains the signal.
                shape: (n_batch, n_vertices, input_dim)
            The second tensor contains the barycentric coordinates and parallel transport angles.
                shape: (n_batch, n_vertices, n_radial, n_angular, 3, 3)

        Returns
        -------
            One tensor. The resulting embeddings.
                shape: (n_batch, n_vertices, output_dim)
        """
        # signals : (n_batch, n_vertices, input_dim)
        # bc      : (n_batch, n_vertices, n_radial, n_angular, 3, 3)
        # angles  : (n_batch, n_vertices, n_vertices)
        signals, bc = inputs

        # Get transported and interpolated feature vectors at each template vertex
        # template_vertex_interpolations : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        template_vertex_interpolations = self._signal_pullback_with_parallel_transport(signals, bc, self.input_types)

        # Reshape vertex signals into their geometric components
        # signals : (n_batch, n_vertices, input_dim / 2, 2)
        signals_shape = tf.shape(signals)
        signals = tf.reshape(signals, (signals_shape[0], signals_shape[1], self.input_dim_halve, 2))
        return self.call_helper(signals, template_vertex_interpolations, return_self_and_neighbor_embeddings=False)

    @tf.function
    def call_helper(self, signals, template_vertex_interpolations, return_self_and_neighbor_embeddings=False):
        """Computes GEM convolution using given signals and template vertex interpolations.

        Parameters
        ----------
        signals: tf.Tensor
            The tensor that contains the signals per vertex.
                shape: (n_batch, n_vertices, input_dim / 2, 2)
        template_vertex_interpolations: tf.Tensor
            The tensor that contains the transported neighbor interpolations.
                shape: (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        return_self_and_neighbor_embeddings: bool
            Whether to return the aggregated embeddings per mesh vertex or merely the embeddings at the convolution
            center and the template vertices.

        Returns
        -------
        tuple:
            Output type depends on 'return_self_and_neighbor_embeddings'. If 'False', this function returns the final
            embedding vectors per mesh vertex.
                shape: (n_batch, n_vertices, output_dim)
            If 'True', this function returns the embedding vectors at the convolution center and the template vertices
            (not aggregated over yet).
                shape: [
                    (n_batch, n_vertices, output_dim / 2, 2),
                    (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
                ]
        """
        ### Compute self-connection embeddings ###
        # 'V_self'    : (output_dim / 2, input_dim / 2, 2)
        # 'K_self'    : (output_dim / 2, input_dim / 2, 2, 2, 2)
        # 'conv_self' : (n_batch, n_vertices, output_dim / 2, 2)
        conv_self = self.get_self_connection_embeddings(signals, self.V_self, self.K_self)

        ### Compute neighbor embeddings ###
        # 'template_vertex_interpolations' : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        # 'V_neigh'                        : (output_dim / 2, input_dim / 2, 4)
        # 'K_neigh'                        : (output_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        if return_self_and_neighbor_embeddings:
            # 'conv_neigh': (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
            conv_neigh = self.get_neigh_embeddings(
                template_vertex_interpolations,
                self.V_neigh,
                self.K_neigh,
                neighbor_aggregation=False
            )
            return conv_self, conv_neigh
        else:
            # 'conv_neigh' : (n_batch, n_vertices, output_dim / 2, 2)
            conv_neigh = self.get_neigh_embeddings(
                template_vertex_interpolations,
                self.V_neigh,
                self.K_neigh,
                neighbor_aggregation=True
            )
            return self.prepare_result(conv_self, conv_neigh)

    @tf.function
    def prepare_result(self, conv_self, conv_neigh):
        """Prepares the result of the convolution.

        Parameters
        ----------
        conv_self: tf.Tensor
            The tensor that contains the embedding vectors for the convolution center.
                shape: (n_batch, n_vertices, output_dim)
        conv_neigh: tf.Tensor
            The tensor that contains the embedding vectors at the template vertices.
                shape: (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)

        Returns
        -------
        tf.Tensor:
            A tensor that contains the aggregated results over the embedding vectors from the convolution centers and
            template vertices. A magnitude activation function has been applied.
                shape: (n_batch, n_vertices, output_dim)
        """
        # Add self-connection and neighbor embeddings for complete conv result
        # conv_self   : (n_batch, n_vertices, output_dim / 2, 2)
        # conv_neigh  : (n_batch, n_vertices, output_dim / 2, 2)
        # result      : (n_batch, n_vertices, output_dim / 2, 2)
        result = conv_self + conv_neigh
        result_shape = tf.shape(result)

        # Apply magnitude activation
        # result_amp : (n_batch, n_vertices, output_dim / 2)
        # result     : (n_batch, n_vertices, output_dim / 2, 2)
        result_amp = tf.maximum(tf.linalg.norm(result, axis=-1), 1e-6)
        result = self.activation_fn(result_amp)[..., None] * tf.math.divide_no_nan(result, result_amp[..., None])

        # Return output in original shape
        # (n_batch, n_vertices, output_dim)
        return tf.reshape(result, (result_shape[0], result_shape[1], self.output_dim_halve * 2))

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
            "input_types": np.array(self.input_types).astype(np.int32).tolist(),
            "output_types": np.array(self.output_types).astype(np.int32).tolist()
        }
        base_config.update(class_config)

        # Prevent double keyword argument (init sets 'include_kernel' to false)
        del base_config["include_kernel"]
        return base_config
