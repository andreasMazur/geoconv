# GeoConv

**Geo**desic **conv**olutional neural networks belong to the category of intrinsic mesh CNNs [1].
They operate directly on object surfaces, therefore expanding the application of convolutions
to non-Euclidean data. The **geoconv** library delivers an implementation of the geodesic convolution [2] 
while taking into account discretization ideas which were given in [3]. Additionally, preprocessing functions, like
computing geodesic polar coordinates on triangulated object meshes [4], which required to feed data into the geodesic
convolution layer are included too.

## Installation
1. Install **[BLAS](https://netlib.org/blas/#_reference_blas_version_3_10_0)** and **[CBLAS](https://netlib.org/blas/#_cblas)**
2. Clone and install **geoconv**:
     ```bash
     git clone https://github.com/andreasMazur/geoconv.git
     cd ./GeodesicConvolution
     pip install -r requirements.txt
     ```
### Hint:

If you do not use this library for preprocessing your data you can safely remove the ``ext_modules``-argument within
``setup.py`` and forget about installing **BLAS** and **CBLAS**. More so, even if you use this library for preprocessing
you can remove this argument. Just set the ``use_c``-flag in the respective preprocessing-function to ``false`` and you
are not required to install **BLAS** and **CBLAS**. However, preprocessing will take longer then.

## Usage

In order to make the usage as simple as possible, we implement the geodesic convolution as a Tensorflow-layer.
Thus, it can be used as any other default Tensorflow layer, making it easy to the users who are familiar 
with Tensorflow.

### Define a GCNN with geoconv:

```python
from tensorflow.keras.layers import InputLayer, Dense
from geoconv.geodesic_conv import ConvGeodesic

import tensorflow as tf


def define_model(signal_shape, barycentric_shape, output_dim, kernel_size=(2, 4), lr=.00045):
    """Define a geodesic convolutional neural network"""
    
    signal_input = InputLayer(shape=signal_shape)
    barycentric = InputLayer(shape=barycentric_shape)
    signal = ConvGeodesic(kernel_size=kernel_size, output_dim=256, amt_kernel=2, activation="relu")([signal, barycentric])
    logits = Dense(output_dim)(signal)

    model = tf.keras.Model(inputs=[signal_input, barycentric], outputs=[logits])
    return model

# Now train/validate/use it like you would with any other tensorflow model..
```

An example training-pipeline is given in ``geoconv.train_examples.train_mpi_faust``

### Inputs and preprocessing

As visible in the minimal example above, the geodesic convolutional layer expects 2 inputs:
1. The signal defined on the mesh vertices (can be anything from descriptors like SHOT [5] to simple 3D-coordinates of
the vertices).
2. Barycentric coordinates for signal interpolation in the format specified by
``geoconv.preprocessing.barycentric_coords.barycentric_coordinates``.

For the latter: **geoconv** supplies you with the necessary preprocessing functions:
1. Use ``geoconv.preprocessing.discrete_gpc.discrete_gpc`` on your triangle meshes (which are given in a format that is
supported by [Trimesh](https://trimsh.org/index.html), e.g. 'ply') to compute discrete local geodesic polar coordinate
systems with the algorithm suggested by [4].
2. Use the GPC-systems and ``geoconv.preprocessing.barycentric_coords.barycentric_coordinates`` to compute the
Barycentric coordinates in the required format. The result can without further effort directly be fed into the layer.

An example preprocessing-pipeline is given in ``geoconv.datasets.mpi_faust.preprocess``.

## Referenced Literature

[1]: Bronstein, Michael M., et al. "Geometric deep learning: Grids, groups, graphs, geodesics, and gauges." 
     arXiv preprint arXiv:2104.13478 (2021).

[2]: Masci, Jonathan, et al. "Geodesic convolutional neural networks on riemannian manifolds." Proceedings of the IEEE
     international conference on computer vision workshops. 2015.

[3]: Poulenard, Adrien, and Maks Ovsjanikov. "Multi-directional geodesic neural networks via equivariant convolution."
     ACM Transactions on Graphics (TOG) 37.6 (2018): 1-14.

[4]: Melvær, Eivind Lyche, and Martin Reimers. "Geodesic polar coordinates on polygonal meshes." Computer Graphics 
     Forum. Vol. 31. No. 8. Oxford, UK: Blackwell Publishing Ltd, 2012.

[5]: Tombari, Federico, Samuele Salti, and Luigi Di Stefano. "Unique signatures of histograms for local surface
     description." European conference on computer vision. Springer, Berlin, Heidelberg, 2010.
