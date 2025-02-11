import unittest
import torch
import tensorflow as tp
import numpy as np

from geoconv.tensorflow.utils.compute_shot_decr import determine_central_values as tf_determine_central_values
from geoconv.tensorflow.utils.compute_shot_decr import shot_descr as tf_shot_descr

from geoconv.pytorch.utils.compute_shot_decr import determine_central_values, shot_descr

class TestComputeShotDecr(unittest.TestCase):
    def test_determine_central_values_same_as_tf(self):
        start = -1.
        stop = 1.
        n_bins = 11

        tf_out = tf_determine_central_values(start, stop, n_bins)
        torch_out = determine_central_values(start, stop, n_bins)

        assert np.allclose(tf_out[0], torch_out[0].numpy())
        assert np.allclose(tf_out[1], torch_out[1])

    def test_shot_descr_same_as_tf(self):
        np.random.seed(1337)
        neighborhoods = np.random.randn(10, 5, 3).astype(dtype=np.float32)
        normals = np.random.randn(10, 3).astype(dtype=np.float32)
        neighborhood_indices = np.random.randint(0, 10, (10, 5), dtype=np.int32)
        radius = np.max(np.linalg.norm(neighborhoods, axis=-1))

        tf_out = tf_shot_descr(neighborhoods, normals, neighborhood_indices, radius)
        torch_out = shot_descr(torch.from_numpy(neighborhoods), torch.from_numpy(normals), torch.from_numpy(neighborhood_indices), radius)

        assert np.allclose(tf_out.numpy(), torch_out.numpy())
    