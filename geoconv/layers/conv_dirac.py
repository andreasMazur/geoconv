from geoconv.layers.conv_intrinsic import ConvIntrinsic

import numpy as np


class ConvDirac(ConvIntrinsic):
    """No interpolation weighting

    This procedure was suggested in:
    > [Multi-directional Geodesic Neural Networks via Equivariant Convolution](https://arxiv.org/abs/1810.02303)
    > Adrien Poulenard and Maks Ovsjanikov
    """

    def define_interpolation_coefficients(self, kernel_matrix):
        """
        Only take the value at ('rho_in', 'theta_in') into account for the patch operator at ('rho_in', 'theta_in')
        """
        interpolation_coefficients = np.zeros(kernel_matrix.shape[:-1] + kernel_matrix.shape[:-1])
        for mean_rho_idx in range(kernel_matrix.shape[0]):
            for mean_theta_idx in range(kernel_matrix.shape[1]):
                interpolation_coefficients[mean_rho_idx, mean_theta_idx, mean_rho_idx, mean_theta_idx] = 1.
        return interpolation_coefficients
