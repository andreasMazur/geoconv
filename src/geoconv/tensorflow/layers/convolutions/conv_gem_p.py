from geoconv.tensorflow.layers.convolutions.conv_gem import ConvGEM, get_kernel_neigh, get_kernel_self
from geoconv.utils.sine_cosine_locs import get_sin_and_cosine_locs_neigh, get_sin_and_cosine_locs_self

import tensorflow as tf


class ConvGEMP(ConvGEM):
    """Implements GEM-CNNs with radially dependent weights and learnable phase offsets."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set in build
        self.sine_and_cosine_locs = None
        self.sine_and_cosine_locs_self = None
        self.phase_weights = None

    def build(self, inputs):
        super().build(inputs)

        ### GEM-CNN+ makes linear coefficients radial-coordinate dependent ###
        self.V_neigh = self.add_weight(
            name="V_neigh",
            shape=(self.output_dim_halve, self.input_dim_halve, 4, self.n_radial),
            trainable=True
        )

        ### 'K_neigh' depends on trainable phase weights in GEM-CNN+ and is hence set in forward pass ###
        del self.K_neigh
        self.sine_and_cosine_locs = tf.constant(
            get_sin_and_cosine_locs_neigh(self.input_types.numpy(), self.output_types.numpy())
        )

        ### 'K_self' depends on trainable phase weights in GEM-CNN+ and is hence set in forward pass ###
        del self.K_self
        self.sine_and_cosine_locs_self = tf.constant(
            get_sin_and_cosine_locs_self(self.input_types.numpy(), self.output_types.numpy(), angle_equals_zero=False)
        )

        ### Define phase weights for creation of neighbor aggr. and self connection kernel matrices ###
        self.phase_weights = self.add_weight(
            name="phase_weights",
            shape=(self.output_dim_halve,),
            trainable=True
        )

    @tf.function(jit_compile=True)
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
        # 'conv_self' : (n_batch, n_vertices, output_dim / 2, 2)
        K_self = get_kernel_self(
            self.input_types,
            self.output_types,
            self.sine_and_cosine_locs_self,
            tf.convert_to_tensor(self.phase_weights)
        )
        conv_self = self.get_self_connection_embeddings(signals, tf.convert_to_tensor(self.V_self), K_self)

        ### Compute neighbor embeddings ###
        # 'template_vertex_interpolations' : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        # 'V_neigh'                        : (output_dim / 2, input_dim / 2, 4)
        # 'input_types'                    : (input_dim / 2, )
        # 'output_types'                   : (output_dim / 2, )
        if return_self_and_neighbor_embeddings:
            # 'conv_neigh': (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
            conv_neigh = self.get_neigh_embeddings_p(
                template_vertex_interpolations,
                tf.convert_to_tensor(self.V_neigh),
                self.input_types,
                self.output_types,
                self.sine_and_cosine_locs,
                tf.convert_to_tensor(self.phase_weights),
                neighbor_aggregation=False
            )
            return conv_self, conv_neigh
        else:
            # 'conv_neigh' : (n_batch, n_vertices, output_dim / 2, 2)
            conv_neigh = self.get_neigh_embeddings_p(
                template_vertex_interpolations,
                tf.convert_to_tensor(self.V_neigh),
                self.input_types,
                self.output_types,
                self.sine_and_cosine_locs,
                tf.convert_to_tensor(self.phase_weights),
                neighbor_aggregation=True
            )
            return self.prepare_result(conv_self, conv_neigh)

    @tf.function(jit_compile=True)
    def get_neigh_embeddings_p(self,
                               interpolations,
                               V_neigh,
                               input_types,
                               output_types,
                               sine_and_cosine_locs,
                               phase_weights,
                               neighbor_aggregation=True):
        """Constructs the neighborhood aggregation kernel matrix and subsequently embeds the input signal.

        Parameters
        ----------
        interpolations: tf.Tensor
            The interpolated feature vectors at the template vertices.
        V_neigh: tf.Tensor
            The trainable weights for the neighbor aggregation.
        input_types: tf.Tensor
            The input types tensor.
        output_types: tf.Tensor
            The output types tensor.
        sine_and_cosine_locs: tf.Tensor
            Tensors that indicate where to put sines and cosines according to input-output pairs in the kernel.
        phase_weights: tf.Tensor
            The phase weights to be used for creating the kernel matrix.
        neighbor_aggregation: bool
            Whether to sum over the neighbor embeddings.

        Returns
        -------
        tf.Tensor:
            Embeddings computed using embedded neighbor features.
        """
        ### Construct the kernel matrix ###
        # 'K_neigh': (output_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        K_neigh = get_kernel_neigh(
            gamma_in=input_types,
            gamma_out=output_types,
            sine_and_cosine_locs=sine_and_cosine_locs,
            angles=self.all_angular_coordinates,
            phase_weights=phase_weights
        )

        ### Determine weight matrix for neighbor aggregation ###
        # 'V_neigh' : (output_dim / 2, input_dim / 2, 4, n_radial)
        # 'K_neigh' : (output_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        # 'W_neigh' : (output_dim / 2, input_dim / 2   , n_radial, n_angular, 2, 2)
        W_neigh = tf.einsum("mnsr,mnsaxy->mnraxy", V_neigh, K_neigh)

        ### Compute neighbor embeddings ###
        # W_neigh        : (output_dim / 2, input_dim / 2, n_radial, n_angular, 2, 2)
        # interpolations : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        if neighbor_aggregation:
            # return : (n_batch, n_vertices, output_dim / 2, 2)
            return tf.einsum("mnraxy,bkrany->bkmx", W_neigh, interpolations)
        else:
            # return : (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
            return tf.einsum("mnraxy,bkrany->bkramx", W_neigh, interpolations)

