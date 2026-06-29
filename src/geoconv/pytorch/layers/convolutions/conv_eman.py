from geoconv.pytorch.layers.convolutions.conv_gem import ConvGEM, get_kernel_neigh
from geoconv.utils.sine_cosine_locs import get_sin_and_cosine_locs_self, get_sin_and_cosine_locs_neigh

from torch import nn

import torch


class ConvEMAN(ConvGEM):
    def __init__(self, attention_types, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ### Set output types for attention mappings (keys and queries) ###
        self.attention_types = torch.tensor(attention_types, dtype=torch.float32)
        self.attention_dim_halve = self.attention_types.shape[0]

        ### Prepare self query kernels and weights ###
        # 'K_self_query': (attention_dim / 2, input_dim / 2, 2, 2, 2)
        self.K_self_query = torch.tensor(
            get_sin_and_cosine_locs_self(gamma_in=self.input_types.numpy(), gamma_out=self.attention_types.numpy())
        )

        # 'V_self_query': (attention_dim / 2, input_dim / 2, 2)
        self.V_self_query = nn.Parameter(torch.empty(int(self.attention_dim_halve), int(self.input_dim_halve), 2))
        nn.init.xavier_uniform_(self.V_self_query)

        ### Prepare self keys kernels and weights ###
        # 'K_self_keys': (attention_dim / 2, input_dim / 2, 2, 2, 2)
        self.K_self_keys = torch.tensor(
            get_sin_and_cosine_locs_self(gamma_in=self.input_types.numpy(), gamma_out=self.attention_types.numpy())
        )

        # 'V_self_keys': (attention_dim / 2, input_dim / 2, 2)
        self.V_self_keys = nn.Parameter(torch.empty(int(self.attention_dim_halve), int(self.input_dim_halve), 2))
        nn.init.xavier_uniform_(self.V_self_keys)

        ### Prepare neighbor kernels and weights ###
        # 'K_neigh_keys': (attention_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        self.K_neigh_keys = get_kernel_neigh(
            gamma_in=self.input_types,
            gamma_out=self.attention_types,
            sine_and_cosine_locs=torch.tensor(
                get_sin_and_cosine_locs_neigh(self.input_types.numpy(), self.attention_types.numpy())
            ),
            angles=self.all_angular_coordinates,
            phase_weights=torch.full(size=(self.output_dim_halve, 1), fill_value=0.)
        )

        # 'V_neigh_keys': (attention_dim / 2, input_dim / 2, 4)
        self.V_neigh_keys = nn.Parameter(
            torch.empty(int(self.attention_dim_halve), int(self.input_dim_halve), 4, int(self.n_radial))
        )

        self.attention_dim_halve = torch.tensor(self.attention_dim_halve, dtype=torch.float32)

    def forward(self, inputs):
        """Computes the equivariant mesh attention convolution

        Parameters
        ----------
        inputs: (torch.Tensor, torch.Tensor)
            The first tensor has shape [n_batch, n_vertices, input_dim] and contains the signals for each vertex. The
            second tensor has shape [n_batch, n_vertices, n_radial, n_angular, 3, 2] and contains the barycentric
            coordinates.

        Returns
        -------
        torch.Tensor
            A tensor of size [n_batch, n_vertices, output_dim], containing the new signal-embeddings for each mesh
            vertex.
        """
        # signals : (n_batch, n_vertices, input_dim)
        # bc      : (n_batch, n_vertices, n_radial, n_angular, 3, 2)
        # angles  : (n_batch, n_vertices, n_vertices)
        signals, bc = inputs

        # Get transported and interpolated feature vectors at each template vertex
        # template_vertex_interpolations : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        template_vertex_interpolations = self._signal_pullback_with_parallel_transport(signals, bc, self.input_types)

        # Reshape vertex signals into their geometric components
        # signals : (n_batch, n_vertices, input_dim / 2, 2)
        signals_shape = signals.size()
        signals = torch.reshape(signals, (signals_shape[0], signals_shape[1], self.input_dim_halve, 2))

        # Calculate attention coefficients
        # A_self  : (n_batch, n_vertices)
        # A_neigh : (n_batch, n_vertices, n_radial, n_angular)
        A_self, A_neigh = self.get_attention_coefficients(signals, template_vertex_interpolations)

        # Calculate self- and neighbor embeddings
        # conv_self  : (n_batch, n_vertices, output_dim / 2, 2)
        # conv_neigh : (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
        conv_self, conv_neigh = super().forward_helper(
            signals, template_vertex_interpolations, return_self_and_neighbor_embeddings=True
        )

        # Multiply attention weight onto self-embeddings
        # conv_self  : (n_batch, n_vertices, output_dim / 2, 2)
        # conv_neigh : (n_batch, n_vertices, output_dim / 2, 2)
        conv_self = torch.einsum("bk,bkmx->bkmx", A_self, conv_self)
        conv_neigh = torch.einsum("bkra,bkramx->bkmx", A_neigh, conv_neigh)
        return self.prepare_result(conv_self, conv_neigh)

    def get_attention_coefficients(self, self_con_signal, template_vertex_interpolations):
        """Computes the attention coefficients for self-connections and neighbor aggregations.

        Parameters
        ----------
        self_con_signal: torch.Tensor
            A tensor of shape [n_batch, n_vertices, input_dim / 2, 2], containing the signals at the self-connections.
        template_vertex_interpolations: torch:tensor
            A tensor of shape [n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2], containing the signals
            at the template vertices.

        Returns
        -------
        (torch.Tensor, torch.Tensor):
            One tensor of shape [n_batch, n_vertices], containing the attention coefficient for the self-connection, and
            a second tensor of shape [n_batch, n_vertices, n_radial, n_angular], containing the attention coefficients
            for the template vertices.
        """
        ### Compute QUERY SELF tensor in preparation for HELPER SELF/NEIGH tensor ###
        # 'self_con_signal' : (n_batch, n_vertices, input_dim / 2, 2)
        # 'V_self_query'    : (attention_dim / 2, input_dim / 2, 2)
        # 'K_self_query'    : (attention_dim / 2, input_dim / 2, 2, 2, 2)
        # 'query_self'      : (n_batch, n_vertices, attention_dim / 2, 2)
        query_self = self.get_self_connection_embeddings(self_con_signal, self.V_self_query, self.K_self_query)

        ### Compute KEYS SELF tensor in preparation for HELPER SELF tensor ###
        # 'self_con_signal' : (n_batch, n_vertices, input_dim / 2, 2)
        # 'V_self_keys'     : (attention_dim / 2, input_dim / 2, 2)
        # 'K_self_keys'     : (attention_dim / 2, input_dim / 2, 2, 2, 2)
        # 'keys_self'       : (n_batch, n_vertices, attention_dim / 2, 2)
        keys_self = self.get_self_connection_embeddings(self_con_signal, self.V_self_keys, self.K_self_keys)

        ### Compute KEYS NEIGH tensor in preparation for HELPER NEIGH tensor ###
        # 'template_vertex_interpolations' : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        # 'V_neigh_keys'                   : (attention_dim / 2, input_dim / 2, 4)
        # 'K_neigh_keys'                   : (attention_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        # 'keys_neigh'                     : (n_batch, n_vertices, n_radial, n_angular, attention_dim / 2, 2)
        keys_neigh = self.get_neigh_embeddings_radially_dependent(
            template_vertex_interpolations, self.V_neigh_keys, self.K_neigh_keys, neighbor_aggregation=False
        )

        ### Log-sum-exp trick for numerical stability ###
        # 'keys_self'  : (n_batch, n_vertices, attention_dim / 2, 2)
        # 'query_self' : (n_batch, n_vertices, attention_dim / 2, 2)
        # 'base_self'  : (n_batch, n_vertices)
        base_self = torch.einsum("bkmx,bkmx->bk", keys_self, query_self) / torch.sqrt(2 * self.attention_dim_halve)

        # 'keys_neigh' : (n_batch, n_vertices, n_radial, n_angular, attention_dim / 2, 2)
        # 'query_self' : (n_batch, n_vertices,                      attention_dim / 2, 2)
        # 'base_neigh' : (n_batch, n_vertices, n_radial, n_angular)
        base_neigh = torch.einsum("bkramx,bkmx->bkra", keys_neigh, query_self) / torch.sqrt(2 * self.attention_dim_halve)

        # 'base_self'  : (n_batch, n_vertices)
        # 'base_neigh' : (n_batch, n_vertices, n_radial, n_angular)
        # 'max_base'   : (n_batch, n_vertices)
        max_base = torch.maximum(base_self, torch.amax(base_neigh, dim=[-2, -1]))

        ### Compute HELPER SELF tensor ###
        # 'base_self'   : (n_batch, n_vertices)
        # 'max_base'    : (n_batch, n_vertices)
        # 'helper_self' : (n_batch, n_vertices)
        helper_self = torch.exp(base_self - max_base)

        ### Compute HELPER NEIGH tensor ###
        # 'base_neigh'   : (n_batch, n_vertices, n_radial, n_angular)
        # 'max_base'     : (n_batch, n_vertices)
        # 'helper_neigh' : (n_batch, n_vertices, n_radial, n_angular)
        helper_neigh = torch.exp(base_neigh - max_base[..., None, None])

        ### Compute denominator for attention coefficients ###
        # 'helper_self'  : (n_batch, n_vertices)
        # 'helper_neigh' : (n_batch, n_vertices, n_radial, n_angular)
        # 'denominator'  : (n_batch, n_vertices)
        denominator = helper_self + torch.einsum("bkra->bk", helper_neigh)

        ### Calculate self-attention coefficients ###
        # 'helper_self' : (n_batch, n_vertices)
        # 'denominator' : (n_batch, n_vertices)
        # 'A_self'      : (n_batch, n_vertices)
        A_self = torch.where(helper_self != 0., helper_self / denominator, 0.)

        ### Calculate neigh-attention coefficients ###
        # 'helper_neigh' : (n_batch, n_vertices, n_radial, n_angular)
        # 'denominator'  : (n_batch, n_vertices)
        # 'A_neigh'      : (n_batch, n_vertices, n_radial, n_angular)
        A_neigh = torch.where(helper_neigh != 0., helper_neigh / denominator[..., None, None], 0.)

        ### Return attention coefficients ###
        # 'A_self'  : (n_batch, n_vertices)
        # 'A_neigh' : (n_batch, n_vertices, n_radial, n_angular)
        return A_self, A_neigh

    def get_neigh_embeddings_radially_dependent(self, interpolations, V_neigh, K_neigh, neighbor_aggregation=True):
        """Constructs the kernel matrix from trainable weights and irreps, and subsequently embeds the input signal.

        Parameters
        ----------
        interpolations: torch.Tensor
            The interpolated feature vectors at the template vertices.
        V_neigh: torch.Tensor
            The trainable weights for the neighbor aggregation.
        K_neigh: torch.Tensor
            The kernel tensor for the neighbor aggregation.
        neighbor_aggregation: bool
            Whether to sum over the neighbor embeddings.

        Returns
        -------
        torch.Tensor:
            Embeddings computed using embedded neighbor features.
        """
        ### Determine weight matrix for neighbor aggregation ###
        # 'V_neigh' : (output_dim / 2, input_dim / 2, 4, n_radial)
        # 'K_neigh' : (output_dim / 2, input_dim / 2, 4, n_angular, 2, 2)
        # 'W_neigh' : (output_dim / 2, input_dim / 2   , n_radial, n_angular, 2, 2)
        W_neigh = torch.einsum("mnsr,mnsaxy->mnraxy", V_neigh, K_neigh)

        ### Compute neighbor embeddings ###
        # W_neigh        : (output_dim / 2, input_dim / 2, n_radial, n_angular, 2, 2)
        # interpolations : (n_batch, n_vertices, n_radial, n_angular, input_dim / 2, 2)
        if neighbor_aggregation:
            # return : (n_batch, n_vertices, output_dim / 2, 2)
            return torch.einsum("mnraxy,bkrany->bkmx", W_neigh, interpolations)
        else:
            # return : (n_batch, n_vertices, n_radial, n_angular, output_dim / 2, 2)
            return torch.einsum("mnraxy,bkrany->bkramx", W_neigh, interpolations)
