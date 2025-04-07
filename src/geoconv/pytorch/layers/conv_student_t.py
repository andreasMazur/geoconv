from geoconv.utils.misc import angle_distance
from geoconv.pytorch.layers import ConvIntrinsic

import torch
from torch.nn.functional import softmax


def gamma_func(x):
    """Computes the gamma value for the given degrees of freedom

    Parameters
    ----------
    x: float
        The input argument for the gamma function of the student-t distribution

    Returns
    -------
    float:
        The gamma value for the given degrees of freedom
    """
    assert x >= 1/2, "You need to have at least one degree of freedom."
    n = torch.floor(x)
    if x - n == 1/2:
        return (torch.jit._builtins.math.factorial(2 * n) / (torch.jit._builtins.math.factorial(n) * 4 ** n)) * torch.sqrt(torch.pi)
    else:
        return torch.jit._builtins.math.factorial(n)


def student_t_pdf(mean_rho, mean_theta, rho, theta, dof):
    """Student-t probability distribution for geodesic polar coordinates

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
    max_angle = torch.maximum(mean_theta, theta)
    min_angle = torch.minimum(mean_theta, theta)
    delta_angle = angle_distance(max_angle, min_angle)
    delta_angle = (1 + ((delta_angle ** 2) / dof)) ** (- (dof + 1) / 2)

    # Compute delta rho
    delta_rho = torch.abs(rho - mean_rho)
    delta_rho = (1 + ((delta_rho ** 2) / dof)) ** (- (dof + 1) / 2)

    quotient = (gamma_func((dof + 1) / 2) / (torch.sqrt(dof * torch.pi) * gamma_func(dof / 2))) ** 2

    return quotient * delta_rho * delta_angle


class ConvStudentT(ConvIntrinsic):
    """Student-t vertex weighting"""
    def __init__(self, *args, dof=2, **kwargs):
        self.dof = dof
        super().__init__(*args, **kwargs)

    def define_kernel_values(self, template_matrix):
        template_matrix[:, :, 0] = template_matrix[:, :, 0] / template_matrix[:, :, 0].max()
        interpolation_coefficients = torch.zeros(template_matrix.shape[:-1] + template_matrix.shape[:-1])
        for mean_rho_idx in range(template_matrix.shape[0]):
            for mean_theta_idx in range(template_matrix.shape[1]):
                mean_rho, mean_theta = template_matrix[mean_rho_idx, mean_theta_idx]
                for rho_idx in range(template_matrix.shape[0]):
                    for theta_idx in range(template_matrix.shape[1]):
                        rho, theta = template_matrix[rho_idx, theta_idx]
                        interpolation_coefficients[mean_rho_idx, mean_theta_idx, rho_idx, theta_idx] = student_t_pdf(
                            mean_rho, mean_theta, rho, theta, self.dof
                        )
                interpolation_coefficients[mean_rho_idx, mean_theta_idx] = softmax(
                    interpolation_coefficients[mean_rho_idx, mean_theta_idx]
                )
        return interpolation_coefficients
