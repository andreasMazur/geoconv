import unittest
import torch
import tensorflow as tf
import numpy as np

from geoconv.pytorch.utils.tensor_utils import tensor_scatter_nd_add_, tensor_scatter_nd_update_, histogram_fixed_width_bins

class TestTensorUtils(unittest.TestCase):
    def test_histogram_fixed_width_bins_same_as_tf(self):
        np.random.seed(1337)
        values = np.random.randn(3040, 4)
        value_range = np.array([-1.0, 1.0])
        n_bins = 11
        tf_out = tf.histogram_fixed_width_bins(values, value_range, n_bins)
        torch_out = histogram_fixed_width_bins(torch.from_numpy(values), torch.from_numpy(value_range), n_bins)

        assert np.allclose(tf_out.numpy(), torch_out.numpy())

    def test_scatter_nd_add_same_as_tf(self):
        np.random.seed(1337)
        # Dims taken from bunny histogram update example
        dest = np.random.randn(3040, 8, 2, 2, 11)
        indices = np.random.randint(0, 2, (3040, 4, 5))
        source = np.random.randn(3040, 4)

        tf_dest = tf.tensor_scatter_nd_add(dest, indices, source)
        torch_dest = torch.from_numpy(dest)
        tensor_scatter_nd_add_(torch_dest, torch.from_numpy(indices), torch.from_numpy(source))

        assert np.allclose(tf_dest.numpy(), torch_dest.numpy(), atol=1e-7)
    
    # def test_scatter_nd_update_same_as_tf(self):
    #     # Dims taken from bunny histogram update example
    #     dest = np.zeros((3040, 8, 2, 2, 11))
    #     indices = np.random.randint(0, 2, (3040, 4, 5))
    #     source = np.random.randn(3040, 4)

    #     tf_dest = tf.tensor_scatter_nd_update(dest, indices, source)
    #     torch_dest = torch.from_numpy(dest)
    #     tensor_scatter_nd_update_(torch_dest, torch.from_numpy(indices), torch.from_numpy(source))

    #     assert np.allclose(tf_dest.numpy(), torch_dest.numpy(), atol=1e-7)
