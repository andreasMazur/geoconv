import unittest
import numpy as np

from geoconv.preprocessing.barycentric_coordinates import create_template_matrix

# TF
import tensorflow as tf
from geoconv.tensorflow.layers.barycentric_coordinates import compute_det as tf_compute_det
from geoconv.tensorflow.layers.barycentric_coordinates import sort_angles as tf_sort_angles
from geoconv.tensorflow.layers.barycentric_coordinates import sort_triangles_ccw as tf_sort_triangles_ccw
from geoconv.tensorflow.layers.barycentric_coordinates import delaunay_condition_check as tf_delaunay_condition_check
from geoconv.tensorflow.layers.barycentric_coordinates import create_all_triangles as tf_create_all_triangles
from geoconv.tensorflow.layers.barycentric_coordinates import compute_interpolation_coefficients as tf_compute_interpolation_coefficients
from geoconv.tensorflow.layers.barycentric_coordinates import compute_interpolation_weights as tf_compute_interpolation_weights
from geoconv.tensorflow.layers.barycentric_coordinates import compute_bc as tf_compute_bc

# PT
import torch
from geoconv.pytorch.layers.barycentric_coordinates import compute_det, sort_angles, sort_triangles_ccw, delaunay_condition_check, create_all_triangles, compute_interpolation_coefficients, compute_interpolation_weights, compute_bc

PROJECTIONS = np.array(
  [[[ 0.00000000e+00,  0.00000000e+00],
    [ 2.64445390e-03, -1.57492876e-03],
    [ 2.71486305e-03,  3.25896218e-03],
    [-5.50807454e-03,  1.31239055e-03],
    [-1.21069630e-03, -5.77326259e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [ 1.32202532e-03, -2.38897931e-03],
    [-9.81319812e-04,  2.88809510e-03],
    [-1.90316013e-03,  2.68686377e-03],
    [ 3.82931856e-03,  3.79324518e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [ 1.51655031e-03, -2.54444196e-03],
    [ 2.85860198e-03,  3.65605019e-03],
    [ 1.68628350e-03,  4.60776640e-03],
    [-5.32799540e-03, -9.49292385e-04]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [-3.39989876e-03, -1.63604110e-03],
    [ 5.66466246e-03, -8.47700692e-04],
    [ 4.98458138e-03,  3.10012116e-03],
    [ 5.97603945e-03,  1.35031028e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [-6.17059926e-03, -1.34209730e-03],
    [ 4.92907641e-03, -6.24534860e-03],
    [-7.59543572e-03,  2.57738773e-03],
    [-8.36110394e-03, -3.42584052e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [-1.12643617e-03,  3.14684119e-03],
    [-3.25755635e-03,  2.58342247e-03],
    [-5.29518025e-03,  9.21387284e-04],
    [-5.35271922e-03, -1.66613318e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [-3.07676266e-03, -8.40830253e-05],
    [ 3.60376248e-03,  1.15962629e-03],
    [ 2.16063485e-03,  4.43958165e-03],
    [-5.61038125e-03, -2.60899728e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [-2.16792594e-03,  5.78882114e-04],
    [-2.46440526e-03,  1.88833044e-04],
    [-5.58413193e-03,  9.60324134e-04],
    [ 4.97922674e-03, -2.99730059e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [-1.44406955e-03,  2.98562879e-03],
    [-3.89363966e-03, -2.72571691e-03],
    [ 1.29127700e-04, -5.78946574e-03],
    [ 5.87642100e-03,  2.08071712e-03]],

    [[ 0.00000000e+00,  0.00000000e+00],
    [ 2.93852645e-03,  2.31820950e-03],
    [ 5.24413632e-03,  1.95451500e-03],
    [-5.45202056e-03, -1.56543439e-03],
    [-5.69314929e-03, -2.53963750e-03]]]
)

class TestUtils(unittest.TestCase):
    def test_create_all_triangles(self):
        tf_out = tf_create_all_triangles(PROJECTIONS)
        torch_out = create_all_triangles(torch.from_numpy(PROJECTIONS).to(torch.float32))

        assert np.allclose(tf_out[0].numpy(), torch_out[0].numpy())
        assert np.allclose(tf_out[1].numpy(), torch_out[1].numpy())

    def test_compute_det_same_as_tf(self):
        np.random.seed(1337)
        batched_matrices = np.random.rand(10, 3, 3)

        dets_expected = tf_compute_det(tf.constant(batched_matrices, dtype=tf.float32)).numpy()
        dets = compute_det(torch.from_numpy(batched_matrices).to(torch.float32)).numpy()

        assert np.allclose(dets, dets_expected)
    def test_sort_angles_same_as_tf(self):
        np.random.seed(1337)
        angles = np.random.rand(10, 3)

        sorted_indices_expected = tf_sort_angles(tf.constant(angles, dtype=tf.float32)).numpy()
        sorted_indices = sort_angles(torch.from_numpy(angles).to(torch.float32)).numpy()

        assert np.allclose(sorted_indices, sorted_indices_expected)

    def test_sort_traingles_ccw_same_as_tf(self):
        tf_tris = tf_create_all_triangles(PROJECTIONS)
        torch_tris = create_all_triangles(torch.from_numpy(PROJECTIONS).to(torch.float32))
        tf_out = tf_sort_triangles_ccw(tf_tris)
        torch_out = sort_triangles_ccw(torch_tris)

        assert np.allclose(tf_out.numpy(), torch_out.numpy())

    def test_delaunay_condition_check_same_as_tf(self):
        tf_tris, _ = tf_create_all_triangles(PROJECTIONS)
        tf_out = tf_delaunay_condition_check(tf_tris, PROJECTIONS)

        torch_tris, _ = create_all_triangles(torch.from_numpy(PROJECTIONS).to(torch.float32))
        torch_out = delaunay_condition_check(torch_tris, torch.from_numpy(PROJECTIONS).to(torch.float32))

        assert np.allclose(tf_out.numpy(), torch_out.numpy())

    def test_compute_interpolation_coefficients_same_as_tf(self):
        np.random.seed(1337)
        template = create_template_matrix(n_radial=5, n_angular=6, radius=5.5389e-02, in_cart=True)

        tf_tris, _ = tf_create_all_triangles(PROJECTIONS)
        tf_out = tf_compute_interpolation_coefficients(tf_tris, template)

        torch_tris, _ = create_all_triangles(torch.from_numpy(PROJECTIONS))
        torch_out = compute_interpolation_coefficients(torch_tris, torch.from_numpy(template).to(torch.float32))

        assert np.allclose(tf_out.numpy(), torch_out.numpy(), atol=1e-7)

    def test_compute_interpolation_weights_same_as_tf(self):
        np.random.seed(1337)
        template = create_template_matrix(n_radial=5, n_angular=6, radius=5.5389e-02, in_cart=True)

        tf_out = tf_compute_interpolation_weights(template, PROJECTIONS)
        torch_out = compute_interpolation_weights(torch.from_numpy(template), torch.from_numpy(PROJECTIONS))

        assert np.allclose(tf_out[0].numpy(), torch_out[0].numpy(), atol=1e-7) # selected_bc
        assert np.all(tf_out[1].numpy() == torch_out[1].numpy()) # selected_indices

    def test_compute_bc_same_as_tf(self):
        np.random.seed(1337)
        template = create_template_matrix(n_radial=5, n_angular=6, radius=5.5389e-02, in_cart=True)

        tf_out = tf_compute_bc(template, PROJECTIONS)
        torch_out = compute_bc(torch.from_numpy(template), torch.from_numpy(PROJECTIONS))

        assert np.allclose(tf_out[0].numpy(), torch_out[0].numpy(), atol=1e-7)
        assert np.allclose(tf_out[1].numpy(), torch_out[1].numpy(), atol=1e-7)