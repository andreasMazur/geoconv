from geoconv.pytorch.layers import (
    AngularMaxPooling,
    ConvDirac,
    ConvEMAN,
    ConvEMANP,
    ConvGEM,
    ConvGEMP,
    ConvGeodesic,
    ConvHarmonic,
)
from geoconv.preprocessing.atlas import load_atlas
from geoconv.utils.parallel_transport import concat_bc_and_angles

from geoconv_examples.mnist.preprocess import preprocess_mnist

import torch
import unittest
import numpy as np
import os


class TestTorchLayers(unittest.TestCase):
    """Exercise the PyTorch counterparts of the TensorFlow layer tests."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Layer configuration
        self.n_radial = 2
        self.n_angular = 4

        # Set path to where MNIST test dataset should be stored
        self.temp_dataset_dir = f"{os.path.dirname(__file__)}/test_mnist_atlas"

        # Create MNIST atlas
        preprocess_mnist(
            output_path=self.temp_dataset_dir,
            max_chart_radius=0.02,
            n_radials=[2],
            n_angulars=[4],
            max_temp_radius=None,
            method="hdm",
            normalization_method="hdm",
            processes=1
        )

        # Load MNIST atlas and template radius
        self.mnist_atlas = load_atlas(f"{self.temp_dataset_dir}.hdf5")
        self.template_radius = [k for k, v in self.mnist_atlas.barycentric_coordinates.items()][-1][-1]
        self.n_vertices = self.mnist_atlas.triangle_mesh.vertices.shape[0]

        # Define layer keyword arguments
        self._layer_kwargs = {
            "feature_input_dim": 2,
            "n_radial": self.n_radial,
            "n_angular": self.n_angular,
            "template_radius": self.template_radius,
            "activation_fn": torch.nn.Identity(),
        }

    def _next_image_and_barycentric_coordinates(self):
        """Returns a random complex-valued signal and the atlas coordinates."""
        # Use a random image instead of the TensorFlow dataset to avoid TensorFlow dependency
        image = np.random.uniform(size=(1, self.n_vertices, 2)).astype(np.float32)
        bc = self.mnist_atlas.barycentric_coordinates[(2, 4, self.template_radius)][None, ...]
        return torch.from_numpy(image).float(), torch.from_numpy(bc).float()

    def _next_image_and_barycentric_coordinates_with_angles(self):
        """Returns a random signal with barycentric coordinates and transport angles."""
        image, _ = self._next_image_and_barycentric_coordinates()

        # Load barycentric coordinates
        bc = self.mnist_atlas.barycentric_coordinates[(2, 4, self.template_radius)]

        # Concat angles (plane has all zero transport angles)
        bc = concat_bc_and_angles(bc, np.zeros((self.n_vertices, self.n_vertices)))[None, ...]
        return image, torch.from_numpy(bc).float()

    def test_isc_forward_pass(self):
        # Define ISC layer
        layer = ConvDirac(
            output_dim=1,
            rotation_delta=1,
            **self._layer_kwargs
        )
        amp = AngularMaxPooling()

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"ISC | output dimension: {tuple(output.shape)}")

            pooled_output = amp(output)
            print(f"ISC + AMP | output dimension: {tuple(pooled_output.shape)}")

        # Check tensor shapes
        self.assertEqual(output.shape, (1, self.n_vertices, self.n_angular, 1))
        self.assertEqual(pooled_output.shape, (1, self.n_vertices, 1))

    def test_gcnn_forward_pass(self):
        # Define GCNN layer
        layer = ConvGeodesic(
            output_dim=1,
            rotation_delta=1,
            **self._layer_kwargs
        )
        amp = AngularMaxPooling()

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"GCNN | output dimension: {tuple(output.shape)}")

            pooled_output = amp(output)
            print(f"GCNN + AMP | output dimension: {tuple(pooled_output.shape)}")

        # Check tensor shapes
        self.assertEqual(output.shape, (1, self.n_vertices, self.n_angular, 1))
        self.assertEqual(pooled_output.shape, (1, self.n_vertices, 1))

    def test_hsn_forward_pass(self):
        # Define HSN layer
        layer = ConvHarmonic(
            output_dim=2,
            rotation_order=1,
            **self._layer_kwargs
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"HSN | output dimension: {tuple(output.shape)}")

        # Check tensor shape
        self.assertEqual(output.shape, (1, self.n_vertices, 2))

    def test_gem_cnn_forward_pass(self):
        # Define GEM-CNN layer
        layer = ConvGEM(
            input_types=[0],
            output_types=[1],
            **self._layer_kwargs
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"GEM-CNN | output dimension: {tuple(output.shape)}")

        # Check tensor shape
        self.assertEqual(output.shape, (1, self.n_vertices, 2))

    def test_gem_p_cnn_forward_pass(self):
        # Define GEM-CNN+ layer
        layer = ConvGEMP(
            input_types=[0],
            output_types=[1],
            **self._layer_kwargs
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"GEM-CNN+ | output dimension: {tuple(output.shape)}")

        # Check tensor shape
        self.assertEqual(output.shape, (1, self.n_vertices, 2))

    def test_eman_forward_pass(self):
        # Define EMAN layer
        layer = ConvEMAN(
            input_types=[0],
            output_types=[1],
            attention_types=[1],
            **self._layer_kwargs
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"EMAN | output dimension: {tuple(output.shape)}")

        # Check tensor shape
        self.assertEqual(output.shape, (1, self.n_vertices, 2))

    def test_eman_p_forward_pass(self):
        # Define EMAN+ layer
        layer = ConvEMANP(
            input_types=[0],
            output_types=[1],
            attention_types=[1],
            **self._layer_kwargs
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        with torch.no_grad():
            output = layer([image, bc])
            print(f"EMAN+ | output dimension: {tuple(output.shape)}")

        # Check tensor shape
        self.assertEqual(output.shape, (1, self.n_vertices, 2))
