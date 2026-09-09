from geoconv.tensorflow.layers import (
    ConvDirac,
    ConvGeodesic,
    ConvHarmonic,
    ConvGEM,
    ConvGEMP,
    ConvEMAN,
    ConvEMANP,
    AngularMaxPooling
)

from geoconv.preprocessing.atlas import load_atlas
from geoconv.utils.parallel_transport import concat_bc_and_angles

from geoconv_examples.mnist.preprocess import preprocess_mnist
from geoconv_examples.mnist.dataset import dataset

import tensorflow as tf
import numpy as np
import unittest
import os


class TestTFLayers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def _next_image_and_barycentric_coordinates(self):
        """Returns the first element of the preprocessed MNIST dataset."""
        mnist_dataset = dataset(
            mnist_atlas=f"{self.temp_dataset_dir}.hdf5",
            set_type="train",
            n_radial=2,
            n_angular=4,
            radius=self.template_radius,
            batch_size=1,
            return_rotations=False
        )
        (image, bc), _ = next(iter(mnist_dataset))
        return image, bc

    def _next_image_and_barycentric_coordinates_with_angles(self):
        """Returns the first element of the preprocessed MNIST dataset, including angles attached to BC."""
        # Load the image
        image, _ = self._next_image_and_barycentric_coordinates()

        # Load barycentric coordinates
        bc = self.mnist_atlas.barycentric_coordinates[(2, 4, self.template_radius)]

        # Concat angles (plane has all zero transport angles)
        bc = concat_bc_and_angles(bc, np.zeros((784, 784)))
        bc = tf.convert_to_tensor(bc[None, ...], dtype=tf.float32)
        return image, bc

    def test_isc_forward_pass(self):
        # Define ISC layer
        layer = ConvDirac(
            template_radius=self.template_radius,
            activation="linear",
            rotation_delta=1,
            output_dim=1
        )
        amp = AngularMaxPooling()

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates()
        output = layer([image, bc])
        print(f"ISC | output dimension: {output.numpy().shape}")

        output = amp(output)
        print(f"ISC + AMP | output dimension: {output.numpy().shape}")

    def test_gcnn_forward_pass(self):
        # Define GCNN layer
        layer = ConvGeodesic(
            template_radius=self.template_radius,
            activation="linear",
            rotation_delta=1,
            output_dim=1
        )
        amp = AngularMaxPooling()

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates()
        output = layer([image, bc])
        print(f"GCNN | output dimension: {output.numpy().shape}")

        output = amp(output)
        print(f"GCNN + AMP | output dimension: {output.numpy().shape}")

    def test_hsn_forward_pass(self):
        # Define HSN layer
        layer = ConvHarmonic(
            output_dim=2,
            rotation_order=1,
            template_radius=self.template_radius,
            activation="linear"
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        output = layer([image, bc])
        print(f"HSN | output dimension: {output.numpy().shape}")

    def test_gem_cnn_forward_pass(self):
        # Define GEM-CNN layer
        layer = ConvGEM(
            input_types=[0],
            output_types=[1],
            template_radius=self.template_radius,
            activation="linear"
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        output = layer([image, bc])
        print(f"GEM-CNN | output dimension: {output.numpy().shape}")

    def test_gem_p_cnn_forward_pass(self):
        # Define GEM-CNN+ layer
        layer = ConvGEMP(
            input_types=[0],
            output_types=[1],
            template_radius=self.template_radius,
            activation="linear"
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        output = layer([image, bc])
        print(f"GEM-CNN+ | output dimension: {output.numpy().shape}")

    def test_eman_forward_pass(self):
        # Define EMAN layer
        layer = ConvEMAN(
            template_radius=self.template_radius,
            activation="linear",
            input_types=[0],
            output_types=[1],
            attention_types=[1]
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        output = layer([image, bc])
        print(f"EMAN | output dimension: {output.numpy().shape}")

    def test_eman_p_forward_pass(self):
        # Define EMAN+ layer
        layer = ConvEMANP(
            template_radius=self.template_radius,
            activation="linear",
            input_types=[0],
            output_types=[1],
            attention_types=[1]
        )

        # Propagate input through layer
        image, bc = self._next_image_and_barycentric_coordinates_with_angles()
        output = layer([image, bc])
        print(f"EMAN+ | output dimension: {output.numpy().shape}")
