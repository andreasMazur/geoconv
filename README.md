# GeoConv

## Let's bend planes to curved surfaces.

<img align="right" style="margin-left: 10px; width: 180px;" src="geoconv_cartoon.png">

Intrinsic mesh CNNs [1] operate directly on object surfaces, therefore expanding the application of convolutions to
non-Euclidean data.

**GeoConv** is a library that provides end-to-end tools for deep learning on surfaces.
That is, whether it is pre-processing your mesh files into a format that can be fed into neural networks, or the
implementation of the **intrinsic surface convolutions** [1] themselves, GeoConv has you covered.

## Implementation

While this library is theoretically motivated by the work of [1], [2] and [3] it also adds additional functionalities
such as the freedom of specifying new kernels, preprocessing algorithms like the one from [4], as well as visualization 
and benchmark tools to verify your layer configuration, your pre-processing results or your trained models.

GeoConv provides the base layer `ConvIntrinsic` as a Tensorflow or Pytorch layer. Both implementations are equivalent.
Only the ways in how they are configured slightly differ due to differences regarding Tensorflow and Pytorch. Check the
minimal example below or the `geoconv_examples`-package for how you configure Intrinsic Mesh CNNs.

## Installation
1. Install **[BLAS](https://netlib.org/blas/#_reference_blas_version_3_10_0)** and **[CBLAS](https://netlib.org/blas/#_cblas)**:
    ```bash
    sudo apt install libatlas-base-dev
    ```

2. Install **geoconv**:
    
    | Installation Variant                 | Command                                                                                 |
    |--------------------------------------|-----------------------------------------------------------------------------------------|
    | GeoConv                              | `pip install geoconv`                                                                   |
    | GeoConv + Tensorflow/Keras (**CPU**) | `pip install geoconv[tensorflow]`                                                       |
    | GeoConv + Tensorflow/Keras (**GPU**) | `pip install geoconv[tensorflow_gpu]`                                                   |
    | GeoConv + Pytorch (**CPU**)          | `pip install geoconv[pytorch] --extra-index-url https://download.pytorch.org/whl/cpu`   |
    | GeoConv + Pytorch (**GPU**)          | `pip install geoconv[pytorch] --extra-index-url https://download.pytorch.org/whl/cu118` |

3. If you want to run the FAUST example you also need to install:
    ```bash
    sudo apt install libflann-dev libeigen3-dev lz4
    pip install cython==0.29.37
    pip install pyshot@git+https://github.com/uhlmanngroup/pyshot@master
    ```

4. In case OpenGL context cannot be created:
    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```

### Minimal Example (TensorFlow)

```python
from geoconv.tensorflow.layers.conv_geodesic import ConvGeodesic
from geoconv.tensorflow.layers.angular_max_pooling import AngularMaxPooling

import keras


def define_model(input_dim, output_dim, n_radial, n_angular):
     """Define a geodesic convolutional neural network"""

     signal_input = keras.layers.InputLayer(shape=(input_dim,))
     barycentric = keras.layers.InputLayer(shape=(n_radial, n_angular, 3, 2))
     signal = ConvGeodesic(
          amt_templates=32,  # 32-dimensional output
          template_radius=0.03,  # maximal geodesic template distance 
          activation="relu",
          rotation_delta=1  # Delta in between template rotations
     )([signal_input, barycentric])
     signal = AngularMaxPooling()(signal)
     logits = keras.layers.Dense(output_dim)(signal)

     model = keras.Model(inputs=[signal_input, barycentric], outputs=[logits])
     return model
```

### Minimal Example (PyTorch)

```python
from geoconv.pytorch.layers.conv_geodesic import ConvGeodesic
from geoconv.pytorch.layers.angular_max_pooling import AngularMaxPooling

from torch import nn


class GCNN(nn.Module):
     def __init__(self, input_dim, output_dim, n_radial, n_angular):
          super().__init__()
          self.geodesic_conv = ConvGeodesic(
               input_shape=[(None, input_dim), (None, n_radial, n_angular, 3, 2)],
               amt_templates=32,  # 32-dimensional output
               template_radius=0.03,  # maximal geodesic template distance 
               activation="relu",
               rotation_delta=1  # Delta in between template rotations
          )
          self.amp = AngularMaxPooling()
          self.output = nn.Linear(in_features=32, out_features=output_dim)

     def forward(self, x):
          signal, barycentric = x
          signal = self.geodesic_conv([signal, barycentric])
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

For the latter: **GeoConv** supplies you with the necessary preprocessing functions:
1. Use ``GPCSystemGroup(mesh).compute(u_max=u_max)`` on your triangle meshes (which are stored in a format that is
supported by **[Trimesh](https://trimsh.org/index.html)**, e.g. 'ply') to compute local geodesic polar coordinate systems with the algorithm
of [4].
2. Use those GPC-systems and ``compute_barycentric_coordinates`` to compute the barycentric coordinates for the kernel 
vertices. The result can without further effort directly be fed into the layer.

**For more thorough explanations on how GeoConv operates check out the `geoconv_examples`-package!**

## Cite

Using my work? Please cite this repository by using the **"Cite this repository"-option** of GitHub
in the right panel.

## Referenced Literature

[1]: Bronstein, Michael M., et al. "Geometric deep learning: Grids, groups, graphs, geodesics, and gauges." 
     arXiv preprint arXiv:2104.13478 (2021).

[2]: Monti, Federico, et al. "Geometric deep learning on graphs and manifolds using mixture model cnns." Proceedings
     of the IEEE conference on computer vision and pattern recognition. 2017.

[3]: Poulenard, Adrien, and Maks Ovsjanikov. "Multi-directional geodesic neural networks via equivariant convolution."
     ACM Transactions on Graphics (TOG) 37.6 (2018): 1-14.


[4]: Melvær, Eivind Lyche, and Martin Reimers. "Geodesic polar coordinates on polygonal meshes." Computer Graphics 
     Forum. Vol. 31. No. 8. Oxford, UK: Blackwell Publishing Ltd, 2012.

[5]: Tombari, Federico, Samuele Salti, and Luigi Di Stefano. "Unique signatures of histograms for local surface
     description." European conference on computer vision. Springer, Berlin, Heidelberg, 2010.
