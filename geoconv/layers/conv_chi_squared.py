from geoconv.layers.conv_intrinsic import ConvIntrinsic
from geoconv.layers.conv_geodesic import angle_distance

import numpy as np


def gamma_func(dof):
    """Computes the gamma value for the given degrees of freedom

    Parameters
    ----------
    dof: int
        Degrees of freedom for chi-squared distribution

    Returns
    -------
    float:
        The gamma value for the given degrees of freedom
    """
    assert dof >= 1, "You need to have at least one degree of freedom."
    r = dof / 2
    if dof == 1/2:
        return np.sqrt(np.pi)
    elif dof == 1:
        return 1.
    else:
        return r * gamma_func(dof - 1)


def chi_squared_pdf(mean_rho, mean_theta, rho, theta, dof):
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
    dof: int
        Degrees of freedom for chi-squared distribution

    Returns
    -------
    float:
        The weight for the interpolation point (rho, theta)
    """
    assert dof >= 1, "You need to have at least one degree of freedom."

    # Compute delta theta
    max_angle = np.maximum(mean_theta, theta)
    min_angle = np.minimum(mean_theta, theta)
    delta_angle = angle_distance(max_angle, min_angle)
    if delta_angle == 0 and dof == 1:
        return 1.
    delta_angle_p = delta_angle ** (dof / 2 - 1)

    # Compute delta rho
    delta_rho = np.abs(rho - mean_rho)
    if delta_rho == 0 and dof == 1:
        return 1.
    delta_rho_p = delta_rho ** (dof / 2 - 1)

    gamma = (1 / (2 ** (dof / 2) * gamma_func(dof))) ** 2

    return gamma * delta_rho_p * delta_angle_p * np.exp(-(delta_rho + delta_angle) / 2)


class ConvChiSquared(ConvIntrinsic):
    """Chi-squared vertex weighting"""
    def __init__(self, *args, dof=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.dof = dof

    def define_interpolation_coefficients(self, kernel_matrix):
        kernel_matrix[:, :, 0] = kernel_matrix[:, :, 0] / kernel_matrix[:, :, 0].max()
        interpolation_coefficients = np.zeros(kernel_matrix.shape[:-1] + kernel_matrix.shape[:-1])
        for mean_rho_idx in range(kernel_matrix.shape[0]):
            for mean_theta_idx in range(kernel_matrix.shape[1]):
                mean_rho, mean_theta = kernel_matrix[mean_rho_idx, mean_theta_idx]
                for rho_idx in range(kernel_matrix.shape[0]):
                    for theta_idx in range(kernel_matrix.shape[1]):
                        rho, theta = kernel_matrix[rho_idx, theta_idx]
                        interpolation_coefficients[mean_rho_idx, mean_theta_idx, rho_idx, theta_idx] = chi_squared_pdf(
                            mean_rho, mean_theta, rho, theta, self.dof
                        )
        return interpolation_coefficients
