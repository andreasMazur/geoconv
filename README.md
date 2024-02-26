# GeoConv

## Let's bend planes to curved surfaces.

Intrinsic mesh CNNs [1] operate directly on object surfaces, therefore expanding the application of convolutions to
non-Euclidean data.

**GeoConv** is a library that provides end-to-end tools for deep learning on surfaces.
That is, whether it is pre-processing your mesh files into a format that can be fed into neural networks, or the
implementation of the **intrinsic surface convolutions** [1] themselves, GeoConv has you covered.

## Background

**Geo**desic **conv**olutional neural networks [2] belong to the category of Intrinsic mesh CNNs. While they portray the
first approach, they are not the only approach to convolution on surfaces. This library implements a **general
parametric framework** for intrinsic surface convolutions, following the ideas of [3] while paying special attention to
the theory for Intrinsic Mesh CNNs described in [1].
That is, while GeoConv provides a theoretically substantiated and elaborated class for the intrinsic surface convolution
(`ConvIntrinsic`), you can easily define new ones by subclassing it. This alleviates you from thinking about the
smallest details of every single aspect which you have to consider when you want to calculate intrinsic surface 
convolutions and thereby allows you to focus on your ideas, that you actually want to realize.

## Implementation

GeoConv provides the base layer `ConvIntrinsic` as a Tensorflow or Pytorch layer. Both implementations are equivalent.
Only the ways in how they are configured slightly differ due to differences regarding Tensorflow and Pytorch. Check the
minimal example below or the examples folder for how you configure Intrinsic Mesh CNNs.

In addition to neural network layers, GeoConv provides you with visualization and benchmark tools to check and verify
your layer configuration, your pre-processing results and your trained models. These tools shall help you to understand 
what happens in every step of the pre-processing pipeline.

## Note

This repository is still in development. It was tested using **Python 3.10.13**.

## Installation
1. Install **[BLAS](https://netlib.org/blas/#_reference_blas_version_3_10_0)** and **[CBLAS](https://netlib.org/blas/#_cblas)**:
     ```bash
     sudo apt install libatlas-base-dev
     ```
2. Install **geoconv**:
     ```bash
     conda create -n geoconv python=3.10.*
     conda activate geoconv
     git clone https://github.com/andreasMazur/geoconv.git
     cd ./geoconv
     pip install -r requirements.txt
     pip install protobuf==3.20.*
     pip install .
     ```

### Hint:

In case you cannot install **BLAS** or **CBLAS** you have the option to remove the ``ext_modules``-argument within
``setup.py``. This will prohibit you from taking advantage of the c-extension module that accelerates GPC-system
computation (required during pre-processing mesh files). This, in turn, implies that you have to set the ``use_c``-flag
in the respective functions to ``False`` such that the Python-alternative implementation is used.

### Minimal Example (TensorFlow)

```python
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling

import tensorflow as tf


def define_model(signal_shape, barycentric_shape, output_dim):
    """Define a geodesic convolutional neural network"""

    signal_input = tf.keras.layers.InputLayer(shape=signal_shape)
    barycentric = tf.keras.layers.InputLayer(shape=barycentric_shape)
    signal = ConvGeodesic(
        amt_templates=32,  # 32-dimensional output
        template_radius=0.03,  # maximal geodesic template distance 
        activation="relu",
        rotation_delta=1  # Delta in between template rotations
    )([signal_input, barycentric])
    signal = AngularMaxPooling()(signal)
    logits = tf.keras.layers.Dense(output_dim)(signal)

    model = tf.keras.Model(inputs=[signal_input, barycentric], outputs=[logits])
    return model
```

### Minimal Example (Pytorch)

```python
from geoconv.pytorch.layers.conv_geodesic import ConvGeodesic
from geoconv.pytorch.layers.angular_max_pooling import AngularMaxPooling

from torch import nn


class GCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.geodesic_conv = ConvGeodesic(
            input_shape=[(None, 3), (None, 5, 8, 3, 2)],  # 3-dimensional signal and 5 x 8 template
            amt_templates=32,  # 32-dimensional output
            template_radius=0.03,  # maximal geodesic template distance 
            activation="relu",
            rotation_delta=1  # Delta in between template rotations
        )
        self.amp = AngularMaxPooling()
        self.output = nn.Linear(in_features=32, out_features=10)
    
    def forward(self, x):
        signal, bc = x
        signal = self.geodesic_conv([signal, bc])
        signal = self.amp(signal)
        return self.output(signal)
```

### Inputs and preprocessing

As visible in the minimal examples above, the intrinsic surface convolutional layer (here geodesic convolution) expects
two inputs:
1. The signal defined on the mesh vertices (can be anything from descriptors like SHOT [5] to simple 3D-coordinates of
the vertices).
2. Barycentric coordinates for signal interpolation in the format specified by the output of
``compute_barycentric_coordinates``.

For the latter: **geoconv** supplies you with the necessary preprocessing functions:
1. Use ``GPCSystemGroup(mesh).compute(u_max=u_max)`` on your triangle meshes (which are stored in a format that is
supported by **[Trimesh](https://trimsh.org/index.html)**, e.g. 'ply') to compute local geodesic polar coordinate systems with the algorithm
of [4].
2. Use those GPC-systems and ``compute_barycentric_coordinates`` to compute the barycentric coordinates for the kernel 
vertices. The result can without further effort directly be fed into the layer.

**For more thorough explanations on how GeoConv operates check out the `examples`-folder!**

## Cite

Using my work? Please cite this repository by using the **"Cite this repository"-option** of GitHub
in the right panel.

## Referenced Literature

[1]: Bronstein, Michael M., et al. "Geometric deep learning: Grids, groups, graphs, geodesics, and gauges." 
     arXiv preprint arXiv:2104.13478 (2021).

[2]: Masci, Jonathan, et al. "Geodesic convolutional neural networks on riemannian manifolds." Proceedings of the IEEE
     international conference on computer vision workshops. 2015.

[3]: Monti, Federico, et al. "Geometric deep learning on graphs and manifolds using mixture model cnns." Proceedings
     of the IEEE conference on computer vision and pattern recognition. 2017.

[4]: Melvær, Eivind Lyche, and Martin Reimers. "Geodesic polar coordinates on polygonal meshes." Computer Graphics 
     Forum. Vol. 31. No. 8. Oxford, UK: Blackwell Publishing Ltd, 2012.

[5]: Tombari, Federico, Samuele Salti, and Luigi Di Stefano. "Unique signatures of histograms for local surface
     description." European conference on computer vision. Springer, Berlin, Heidelberg, 2010.
