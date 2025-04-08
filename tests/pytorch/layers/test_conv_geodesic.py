import unittest
import numpy as np
import torch
import tensorflow as tf
from geoconv.pytorch.layers import ConvGeodesic
from geoconv.tensorflow.layers import ConvGeodesic as TFConvGeodesic

import os

os.environ.update({"CUDA_VISIBLE_DEVICES": "0"})  # Run all on cpu


class TestConvGeodesicSameOutput(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.n_vertices = 10
        self.n_radial = 3
        self.n_angular = 4
        self.n_features = 1
        self.rs = np.random.RandomState(5829201)

        # Generate signal
        self.signal = self.rs.randn(self.batch_size, self.n_vertices, self.n_features)
        weights = self.rs.randn(
            self.batch_size, self.n_vertices, self.n_radial, self.n_angular, 3
        )
        weights /= np.sum(weights, axis=-1, keepdims=True)

        base_indices = np.array([0, 1, 2])
        indices_shuffled = np.empty_like(weights)
        weights_shuffled = np.empty_like(weights)

        for b in range(self.batch_size):
            for v in range(self.n_vertices):
                for r in range(self.n_radial):
                    for a in range(self.n_angular):
                        perm = np.random.permutation(3)
                        indices_shuffled[b, v, r, a] = base_indices[perm]
                        weights_shuffled[b, v, r, a] = weights[b, v, r, a][perm]
        self.bc = np.stack([indices_shuffled, weights_shuffled], axis=-1)

        self.bc = self.bc.astype(np.float32)
        self.signal = self.signal.astype(np.float32)

        # Initialize layers
        self.torch_layer = ConvGeodesic(
            amt_templates=128,
            template_radius=0.7,
            activation="elu",
            rotation_delta=1,
            input_shape=(
                (self.batch_size, self.n_vertices, self.n_features),
                (self.batch_size, self.n_vertices, self.n_radial, self.n_angular, 3, 2),
            ),
        )
        self.tf_layer = TFConvGeodesic(
            amt_templates=128, template_radius=0.7, activation="elu", rotation_delta=1
        )
        self.tf_layer((self.signal, self.bc))

        # Assign weights
        for i, params in enumerate(self.torch_layer.parameters()):
            if i < 2:
                self.tf_layer.weights[i].assign(params.detach().numpy())
            else:
                self.tf_layer.weights[i].assign(params.detach().numpy().reshape(-1))

    def test_weights_match(self):
        for tfp, ptp in zip(self.tf_layer.weights, self.torch_layer.parameters()):
            self.assertTrue(
                np.allclose(tfp.numpy().reshape(-1), ptp.detach().numpy().reshape(-1)),
                "Weights mismatch!",
            )

    def test_kernel_match(self):
        self.assertTrue(
            np.allclose(
                self.tf_layer._template_vertices.numpy(),
                self.torch_layer._template_vertices.numpy(),
            ),
            "Template vertices mismatch!",
        )

        self.assertTrue(
            np.allclose(
                self.tf_layer._kernel.numpy(),
                self.torch_layer._kernel.numpy(),
                atol=1e-5,
            ),
            "Kernel compute mismatch!",
        )

    def test_forward_preproc(self):
        pt_signal = torch.from_numpy(self.signal)
        pt_conv_center = torch.einsum(
            "tef,skf->sket", self.torch_layer._template_self_weights, pt_signal
        )
        tf_conv_center = tf.einsum(
            "tef,skf->sket", self.tf_layer._template_self_weights, self.signal
        )
        self.assertTrue(
            np.allclose(pt_conv_center.detach().numpy(), tf_conv_center.numpy()),
            "Conv center mismatch!",
        )

    def test_signal_pullback(self):
        pt_signal = torch.from_numpy(self.signal)
        pt_bc = torch.from_numpy(self.bc)
        pt_signal_pb = self.torch_layer._signal_pullback(pt_signal, pt_bc)
        tf_signal_pb = self.tf_layer._signal_pullback(self.signal, self.bc)
        self.assertTrue(
            np.allclose(pt_signal_pb.detach().numpy(), tf_signal_pb.numpy()),
            "Signal pullback mismatch!",
        )

    def test_patch_operator(self):
        pt_signal = torch.from_numpy(self.signal)
        pt_bc = torch.from_numpy(self.bc)
        pt_patch_op = self.torch_layer._patch_operator(pt_signal, pt_bc)
        tf_patch_op = self.tf_layer._patch_operator(self.signal, self.bc)
        self.assertTrue(
            np.allclose(pt_patch_op.detach().numpy(), tf_patch_op.numpy(), atol=1e-5),
            "Patch operator mismatch!",
        )

    def test_forward_pass(self):
        pt_signal = torch.from_numpy(self.signal)
        pt_bc = torch.from_numpy(self.bc)
        pt_out = self.torch_layer(pt_signal, pt_bc)
        tf_out = self.tf_layer((self.signal, self.bc))
        self.assertTrue(
            np.allclose(pt_out.detach().numpy(), tf_out.numpy(), atol=1e-5),
            "Forward pass output mismatch!",
        )


if __name__ == "__main__":
    unittest.main()
