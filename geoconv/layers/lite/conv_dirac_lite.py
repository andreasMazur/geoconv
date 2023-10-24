from geoconv.layers.lite.conv_intrinsic_lite import ConvIntrinsicLite

import numpy as np


class ConvDiracLite(ConvIntrinsicLite):
    """No interpolation weighting

    This procedure was suggested in:
    > [Multi-directional Geodesic Neural Networks via Equivariant Convolution](https://arxiv.org/abs/1810.02303)
    > Adrien Poulenard and Maks Ovsjanikov
    """

    def define_interpolation_coefficients(self, template_matrix):
        """
        Only take the value at ('rho_in', 'theta_in') into account for the patch operator at ('rho_in', 'theta_in')
        """
        interpolation_coefficients = np.zeros(template_matrix.shape[:-1] + template_matrix.shape[:-1])
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                interpolation_coefficients[mean_rho_idx, mean_theta_idx, mean_rho_idx, mean_theta_idx] = 1.
        return interpolation_coefficients
