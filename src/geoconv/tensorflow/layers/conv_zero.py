from geoconv.tensorflow.layers.conv_intrinsic import ConvIntrinsic

import numpy as np


class ConvZero(ConvIntrinsic):
    """No interpolation weighting"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_kernel_values(self, template_matrix):
        """Returns all-zero interpolation coefficients, which causes the layer to only work with self-connections."""

        return np.zeros(template_matrix.shape[:-1] + template_matrix.shape[:-1])
