from geoconv.pytorch.layers import ConvIntrinsic
from geoconv.utils.misc import angle_distance

from torch.nn.functional import softmax

import torch


def normal_pdf(mean_rho, mean_theta, var_rho, var_theta, rho, theta):
    """Normal probability distribution for geodesic polar coordinates

    Parameters
    ----------
    mean_rho: float
        Mean radial distance of the normal
    mean_theta: float
        Mean angle for the Gaussian
    var_rho: float
        Mean radial distance variance of the kernel vertices
    var_theta: float
        Mean angle variance of the kernel vertices
    rho: float
        Radial coordinate of the interpolation point that shall be weighted
    theta: float
        Angular coordinate of the interpolation point that shall be weighted

    Returns
    -------
    float:
        The weight for the interpolation point (rho, theta)
    """
    norm_coefficient = 1 / torch.sqrt((2 * torch.pi) ** 2 * var_rho * var_theta)
    max_angle = torch.maximum(mean_theta, theta)
    min_angle = torch.minimum(mean_theta, theta)
    delta_angle = angle_distance(max_angle, min_angle)
    vec = torch.tensor([rho - mean_rho, delta_angle], dtype=rho.dtype, device=rho.device)
    mat = torch.tensor([[1. / var_rho, 0.], [0., 1. / var_theta]], dtype=rho.dtype, device=rho.device)
    exp = torch.exp(-(1 / 2) * vec.T @ mat @ vec)
    return norm_coefficient * exp


class ConvGeodesic(ConvIntrinsic):
    """The geodesic convolutional layer

    Paper, that introduced the geodesic convolution:
    > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
    > Jonathan Masci and Davide Boscaini et al.
    """

    def define_kernel_values(self, template_matrix):
        interpolation_coefficients = torch.zeros(template_matrix.shape[:-1] + template_matrix.shape[:-1], dtype=torch.float32)
        var_rho = template_matrix[:, :, 0].var()
        var_theta = template_matrix[:, :, 1].var()
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                mean_rho, mean_theta = template_matrix[mean_rho_idx, mean_theta_idx]
                for rho_idx in range(template_matrix.shape[0]):
                    for theta_idx in range(template_matrix.shape[1]):
                        rho, theta = template_matrix[rho_idx, theta_idx]
                        interpolation_coefficients[mean_rho_idx, mean_theta_idx, rho_idx, theta_idx] = normal_pdf(
                            mean_rho, mean_theta, var_rho, var_theta, rho, theta
                        )
                interpolation_coefficients[mean_rho_idx, mean_theta_idx] = softmax(
                    interpolation_coefficients[mean_rho_idx, mean_theta_idx]
                )
        return interpolation_coefficients
