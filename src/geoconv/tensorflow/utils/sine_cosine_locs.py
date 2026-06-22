import numpy as np


#################################################################
# Definition for block matrices divided in sine and cosine parts
#################################################################

#### For neighborhood aggregation (4, 2, 2) ####
### Zero to zero ###
ZERO_TO_ZERO = np.array(
    [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
### N to zero ###
SIN_N_TO_ZERO = np.array(
    [[[0.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
COS_N_TO_ZERO = np.array(
    [[[1.0, 0.0], [0.0, 0.0]], [[0.0, -1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
### Zero to M ###
SIN_ZERO_TO_M = np.array(
    [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
COS_ZERO_TO_M = np.array(
    [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
### N to M ###
SIN_N_TO_M = np.array(
    [[[0.0, -1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, 1.0]]]
)
COS_N_TO_M = np.array(
    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [-1.0, 0.0]], [[1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [1.0, 0.0]]]
)

#### For self connections (2, 2, 2) ####
### Zero to zero ###
SELF_SIN_ZERO_TO_ZERO = np.array(  # sin(0) = 0
    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
SELF_COS_ZERO_TO_ZERO = np.array(  # cos(0) = 1
    [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
### N to N ###
SELF_SIN_N_TO_N = np.array(  # sin(\beta)
    [[[0.0, -1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
)
SELF_COS_N_TO_N = np.array(  # cos(\beta)
    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [-1.0, 0.0]]]
)
### N to M ###
SELF_N_TO_M = np.array(  # sin(0) = 0 / cos(0) = 1 but both multiply with 0-matrices
    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
)
#################################################################


def get_sin_and_cosine_locs_neigh(gamma_in, gamma_out):
    """Constructs a tensor that contains 1 and zeros, depending on where sines and cosines shall be.

    Parameters
    ----------
    gamma_in: np.ndarray
        A tensor of shape (d_in,) which contains the input types.
    gamma_out: np.ndarray
        A tensor of shape (d_out,) which contains the output types.

    Returns
    -------
    np.ndarray:
        A tensor of shape (2, d_out, d_in, 4, 2, 2) indicating the sine and cosine locations.
    """
    proj_tensor = np.zeros((2, gamma_out.shape[0], gamma_in.shape[0], 4, 2, 2))
    for gi_idx, input_type in enumerate(gamma_in):
        for go_idx, output_type in enumerate(gamma_out):
            if input_type == 0 and output_type == 0:
                proj_tensor[0, go_idx, gi_idx] = np.array(ZERO_TO_ZERO)
                proj_tensor[1, go_idx, gi_idx] = np.array(ZERO_TO_ZERO)
            elif input_type != 0 and output_type == 0:
                proj_tensor[0, go_idx, gi_idx] = np.array(SIN_N_TO_ZERO)
                proj_tensor[1, go_idx, gi_idx] = np.array(COS_N_TO_ZERO)
            elif input_type == 0 and output_type != 0:
                proj_tensor[0, go_idx, gi_idx] = np.array(SIN_ZERO_TO_M)
                proj_tensor[1, go_idx, gi_idx] = np.array(COS_ZERO_TO_M)
            elif input_type != 0 and output_type != 0:
                proj_tensor[0, go_idx, gi_idx] = np.array(SIN_N_TO_M)
                proj_tensor[1, go_idx, gi_idx] = np.array(COS_N_TO_M)
            else:
                raise ValueError(
                    f"Please select valid in- and output types instead of: in {input_type}, out {output_type}."
                )
    return proj_tensor.astype(np.float32)


def get_sin_and_cosine_locs_self(gamma_in, gamma_out, angle_equals_zero=True):
    """Constructs an array that contains ones and zeros, depending on where sines and cosines shall be.

    Parameters
    ----------
    gamma_in: np.ndarray
        A tensor of shape (d_in,) which contains the output types.
    gamma_out: np.ndarray
        A tensor of shape (d_out,) which contains the input types.
    angle_equals_zero: bool
        Entering an angle of a=0 into the sin(a) and cos(a) of the sines and cosines matrices exactly yields the
        cosine locations. Those are used for the construction of self-connection basis kernels of regular GEM-CNNs
        and EMANs. In contrast, GEM-CNNs+ and EMANs+ require the full sine and cosine locations as they additionally
        train a learnable phase offset within the sines and cosines.

    Returns
    -------
    np.ndarray:
        A tensor of shape (x=2, d_out, d_in, s=2, 2, 2) indicating the sine and cosine locations for the construction of
        the basis kernel tensor 'K_self'. The dimension 'x' refers to the fact that we place sines and cosines. The two
        dimensions 'd_out' and 'd_in' refer to the amount of complex output and input features. The dimension 's' refers
        to the fact that 'K_self' can be constructed of up to 2 basis kernels. The last two dimensions refer to the 2x2
        shape of basis kernels.
    """
    # 'proj_tensor' : (d_out, d_in, 2, 2, 2)
    proj_tensor = np.zeros((2, gamma_out.shape[0], gamma_in.shape[0], 2, 2, 2))
    for gi_idx, input_type in enumerate(gamma_in):
        for go_idx, output_type in enumerate(gamma_out):
            if input_type == 0 and output_type == 0:
                proj_tensor[0, go_idx, gi_idx] = np.array(SELF_SIN_ZERO_TO_ZERO)
                proj_tensor[1, go_idx, gi_idx] = np.array(SELF_COS_ZERO_TO_ZERO)
            elif input_type == output_type:
                proj_tensor[0, go_idx, gi_idx] = np.array(SELF_SIN_N_TO_N)
                proj_tensor[1, go_idx, gi_idx] = np.array(SELF_COS_N_TO_N)
            elif input_type != output_type:
                proj_tensor[0, go_idx, gi_idx] = np.array(SELF_N_TO_M)
                proj_tensor[1, go_idx, gi_idx] = np.array(SELF_N_TO_M)
            else:
                raise ValueError(
                    f"Please select valid in- and output types instead of: in {input_type}, out {output_type}."
                )
    if angle_equals_zero:
        return proj_tensor.astype(np.float32)[1]
    else:
        return proj_tensor.astype(np.float32)
