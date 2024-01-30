from geoconv.layers.conv_intrinsic import ConvIntrinsic

import numpy as np


class ConvZero(ConvIntrinsic):
    """No interpolation weighting"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_interpolation_coefficients(self, template_matrix):
        """Returns all-zero interpolation coefficients."""

        return np.zeros(template_matrix.shape[:-1] + template_matrix.shape[:-1])
