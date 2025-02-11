import unittest

import numpy as np

import tensorflow as tf
from geoconv.tensorflow.utils.compute_shot_lrf import compute_distance_matrix as tf_compute_distance_matrix
from geoconv.tensorflow.utils.compute_shot_lrf import disambiguate_axes as tf_disambiguate_axes
from geoconv.tensorflow.utils.compute_shot_lrf import shot_lrf as tf_shot_lrf
from geoconv.tensorflow.utils.compute_shot_lrf import knn_shot_lrf as tf_knn_shot_lrf
from geoconv.tensorflow.utils.compute_shot_lrf import logarithmic_map as tf_logarithmic_map

import torch
from geoconv.pytorch.utils.compute_shot_lrf import compute_distance_matrix 
from geoconv.pytorch.utils.compute_shot_lrf import disambiguate_axes
from geoconv.pytorch.utils.compute_shot_lrf import shot_lrf, knn_shot_lrf, logarithmic_map

def is_equivalent_eigen_vectors(evs0, evs1):
    matches = np.all(np.isclose(evs0, evs1), axis=1)
    neg_matches = np.all(np.isclose(evs0, -evs1), axis=1)
    return np.logical_or(matches, neg_matches).all()

class TestComputeDistanceMatrix(unittest.TestCase):
    def test_compute_distance_matrix_basic(self):
        vertices = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        expected_output = torch.Tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.4142135], [1.0, 1.4142135, 0.0]])
        output = compute_distance_matrix(vertices)
        torch.testing.assert_close(output, expected_output)

    def test_compute_distance_matrix_same_as_tf(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tf_output = tf_compute_distance_matrix(vertices)
        torch_output = compute_distance_matrix(torch.from_numpy(vertices))
        assert np.allclose(tf_output.numpy(), torch_output.numpy())
    
    def test_compute_distance_matrix_with_nan(self):
        vertices = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float('nan'), 1.0, 0.0]])
        expected_output = torch.Tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        output = compute_distance_matrix(vertices)
        torch.testing.assert_close(output, expected_output)


class TestDisambiguateAxes(unittest.TestCase):
    def test_disambiguate_axes_mixed(self):
        torch.manual_seed(1337)
        neighborhood_vertices = torch.randn((10, 3, 3))
        eigen_vectors = torch.randn((10, 3, 3))
        eigen_vectors /= torch.norm(eigen_vectors, dim=-1, keepdim=True)

        output = disambiguate_axes(neighborhood_vertices, eigen_vectors[:, :, 0])
        assert output.shape == eigen_vectors[:, :, 0].shape
        assert is_equivalent_eigen_vectors(output.numpy(), eigen_vectors[:, :, 0].numpy())
    
    def test_disambiguate_axes_same_as_tf(self):
        np.random.seed(1337)
        neighborhood_vertices = np.random.randn(10, 3, 3)
        eigen_vectors = np.random.randn(10, 3, 3)
        eigen_vectors /= np.linalg.norm(eigen_vectors, axis=-1, keepdims=True)

        tf_output = tf_disambiguate_axes(neighborhood_vertices, eigen_vectors[:, :, 0])
        torch_output = disambiguate_axes(torch.from_numpy(neighborhood_vertices), torch.from_numpy(eigen_vectors[:, :, 0]))

        assert is_equivalent_eigen_vectors(tf_output.numpy(), torch_output.numpy())

class TestShotLRF(unittest.TestCase):
    def test_shot_lrf_same_as_tf(self):
        np.random.seed(1337)
        k_neighbors = 5
        neighborhood = np.random.randn(10, k_neighbors, 3)
        radii = np.random.randn(10)

        tf_output = tf_shot_lrf(neighborhood, radii)
        torch_output = shot_lrf(torch.from_numpy(neighborhood), torch.from_numpy(radii))

        assert np.allclose(tf_output.numpy(), torch_output.numpy())

class TestKnnShotLRF(unittest.TestCase):
    def test_knn_shot_lrf_same_as_tf(self):
        np.random.seed(1337)
        k_neighbors = 5
        vertices = np.random.randn(10, 3).astype(np.float32)
        repetitions = 4

        tf_lrfs, tf_neighborhoods, tf_neighborhood_indices = tf_knn_shot_lrf(k_neighbors, vertices, repetitions)
        torch_lrfs, torch_neighborhoods, torch_neighborhood_indices = knn_shot_lrf(k_neighbors, torch.from_numpy(vertices), repetitions)

        matches = np.transpose(np.isclose(tf_lrfs.numpy(), torch_lrfs.numpy(), atol=1e-5), [0, 2, 1])
        neg_matches = np.transpose(np.isclose(tf_lrfs.numpy(), -torch_lrfs.numpy(), atol=1e-5), [0, 2, 1])

        assert np.all(np.logical_or(matches, neg_matches))
        assert np.allclose(torch_neighborhoods.numpy(), tf_neighborhoods.numpy())
        assert np.allclose(torch_neighborhood_indices.numpy(), tf_neighborhood_indices.numpy())

class TestLogarithmicMap(unittest.TestCase):
    def test_logarithmic_map_same_as_tf(self):
        np.random.seed(1337)
        lrfs = np.random.randn(10, 3, 3).astype(np.float32)
        neighborhoods = np.random.randn(10, 5, 3).astype(np.float32)

        tf_output = tf_logarithmic_map(lrfs, neighborhoods)
        torch_output = logarithmic_map(torch.from_numpy(lrfs), torch.from_numpy(neighborhoods))

        assert np.allclose(tf_output.numpy(), torch_output.numpy(), atol=1e-5)
