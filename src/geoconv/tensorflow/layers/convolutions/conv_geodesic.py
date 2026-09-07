from geoconv.tensorflow.layers import ConvIntrinsic
from geoconv.utils.misc import normal_pdf

import numpy as np
import scipy as sp


class ConvGeodesic(ConvIntrinsic):
    """The geodesic convolutional layer.

    Original paper:
    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
    > Jonathan Masci and Davide Boscaini et al.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the object.

        Parameters
        ----------
        *args: tuple
            The args.
        **kwargs: dict
            The kwargs.
        """
        kwargs.pop("include_kernel", None)
        super().__init__(include_kernel=True, *args, **kwargs)

    def define_kernel_values(self, template_matrix):
        """Defines the kernel values for each template vertex.

        Parameters
        ----------
        template_matrix: np.ndarray
            The template coordinates array of size 'n_radial x n_angular x 2'.

        Returns
        -------
        tf.Tensor
            The interpolation values of the Dirac weighting function.
        """
        interpolation_coefficients = np.zeros(
            template_matrix.shape[:-1] + template_matrix.shape[:-1]
        )
        var_rho = template_matrix[:, :, 0].var()
        var_theta = template_matrix[:, :, 1].var()
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                mean_rho, mean_theta = template_matrix[mean_rho_idx, mean_theta_idx]
                for rho_idx in range(template_matrix.shape[0]):
                    for theta_idx in range(template_matrix.shape[1]):
                        rho, theta = template_matrix[rho_idx, theta_idx]
                        interpolation_coefficients[
                            mean_rho_idx, mean_theta_idx, rho_idx, theta_idx
                        ] = normal_pdf(
                            mean_rho, mean_theta, var_rho, var_theta, rho, theta
                        )
                interpolation_coefficients[mean_rho_idx, mean_theta_idx] = (
                    sp.special.softmax(
                        interpolation_coefficients[mean_rho_idx, mean_theta_idx]
                    )
                )
        return interpolation_coefficients
