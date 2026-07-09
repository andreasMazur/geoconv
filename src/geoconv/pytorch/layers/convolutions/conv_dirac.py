from geoconv.pytorch.layers.convolutions.conv_intrinsic import ConvIntrinsic

import numpy as np


class ConvDirac(ConvIntrinsic):
    def __init__(self, *args, **kwargs):
        kwargs.pop("include_kernel", None)
        super().__init__(include_kernel=False, *args, **kwargs)

    def define_kernel_values(self, template_matrix):
        """Only take the value at ('rho_in', 'theta_in') into account for the patch operator at ('rho_in', 'theta_in').

        [DEPRECATED]:
        This function is only implemented for visualization purposes. During the intrinsic surface convolution,
        interpolation coefficients of the Dirac prior do not need to be used as they do not alter the signal at the
        template vertices.
        """
        interpolation_coefficients = np.zeros(
            template_matrix.shape[:-1] + template_matrix.shape[:-1]
        )
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                interpolation_coefficients[mean_rho_idx, mean_theta_idx, mean_rho_idx, mean_theta_idx] = 1.0
        return interpolation_coefficients
