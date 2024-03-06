from geoconv.tensorflow.layers.conv_intrinsic import ConvIntrinsic
from geoconv.tensorflow.layers.conv_geodesic import angle_distance

import numpy as np
import scipy as sp


def exp_pdf(mean_rho, mean_theta, rho, theta, exp_lambda):
    """Exponential probability distribution for geodesic polar coordinates

    Parameters
    ----------
    mean_rho: float
        Mean radial distance of the normal
    mean_theta: float
        Mean angle for the Gaussian
    rho: float
        Radial coordinate of the interpolation point that shall be weighted
    theta: float
        Angular coordinate of the interpolation point that shall be weighted
    exp_lambda: float
        The lambda parameter for the exponential probability density function

    Returns
    -------
    float:
        The weight for the interpolation point (rho, theta)
    """

    # Compute delta theta
    max_angle = np.maximum(mean_theta, theta)
    min_angle = np.minimum(mean_theta, theta)
    delta_angle = angle_distance(max_angle, min_angle)

    # Compute delta rho
    delta_rho = np.abs(rho - mean_rho)

    return exp_lambda ** 2 * np.exp(-exp_lambda * (delta_rho + delta_angle))


class ConvExp(ConvIntrinsic):
    """Exponential vertex weighting"""
    def __init__(self, *args, exp_lambda=1, **kwargs):
        self.exp_lambda = exp_lambda
        super().__init__(*args, **kwargs)

    def define_kernel_values(self, template_matrix):
        template_matrix[:, :, 0] = template_matrix[:, :, 0] / template_matrix[:, :, 0].max()
        interpolation_coefficients = np.zeros(template_matrix.shape[:-1] + template_matrix.shape[:-1])
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                mean_rho, mean_theta = template_matrix[mean_rho_idx, mean_theta_idx]
                for rho_idx in range(template_matrix.shape[0]):
                    for theta_idx in range(template_matrix.shape[1]):
                        rho, theta = template_matrix[rho_idx, theta_idx]
                        interpolation_coefficients[mean_rho_idx, mean_theta_idx, rho_idx, theta_idx] = exp_pdf(
                            mean_rho, mean_theta, rho, theta, self.exp_lambda
                        )
                interpolation_coefficients[mean_rho_idx, mean_theta_idx] = sp.special.softmax(
                    interpolation_coefficients[mean_rho_idx, mean_theta_idx]
                )
        return interpolation_coefficients
