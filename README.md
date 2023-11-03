# GeoConv

## Let's bend planes to curved surfaces.

Intrinsic mesh CNNs [1] operate directly on object surfaces, therefore expanding the application of convolutions to
non-Euclidean data.

**GeoConv** is a library that provides end-to-end tools for deep learning on surfaces.
That is, whether it is pre-processing your mesh files into a format that can be fed into neural networks, or the
implementation of the **intrinsic surface convolutions** [1] themselves, GeoConv has you covered.

## Background

**Geo**desic **conv**olutional neural networks [2] belong to the category of Intrinsic mesh CNNs. While they portray the
first approach, they are not the only approach to convolution on  surfaces. This library implements a **general
parametric framework** for intrinsic surface convolutions, following the formal description of [3].
That is, while GeoConv provides a theoretically substantiated and elaborated class for the
fundamental intrinsic surface convolution (`ConvIntrinsic`), you can easily define new ones by subclassing it. This
alleviates you from thinking about the nitty-gritty details of every single aspect which you have to consider when you
want to calculate intrinsic surface convolutions and allows you to focus on your ideas, that you actually want to
realize.

Since `ConvIntrinsic` is a Tensorflow layer, you benefit from all the advantages that Tensorflow offers such as a being
able to quickly write a training pipeline by configuring and calling Tensorflow's `fit`-function, exporting computations
onto GPUs or TPUs, saving your models in standardized formats and more.

Furthermore, as the topic of pre-processing mesh data can be a bit obscure, GeoConv provides you additionally with
visualization and benchmark tools to check and verify your layer configuration, your pre-processing results and your
trained models.

## Note

This repository is still in development. It was tested using **Python 3.10.11**.

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

## Minimal Example

In order to make the usage as simple as possible, we implement the intrinsic surface convolution as a Tensorflow-layer.
Thus, it can be used as any other default Tensorflow layer, making it easy to the users who are familiar 
with Tensorflow.

### Define a GCNN with geoconv:

```python
from geoconv.layers.conv_geodesic import ConvGeodesic
from geoconv.layers.angular_max_pooling import AngularMaxPooling
from tensorflow import keras


def define_model(signal_shape, barycentric_shape, output_dim):
    """Define a geodesic convolutional neural network"""

    signal_input = keras.layers.InputLayer(shape=signal_shape)
    barycentric = keras.layers.InputLayer(shape=barycentric_shape)
    signal = ConvGeodesic(
        amt_templates=2,
        template_radius=0.028,
        activation="relu"
    )([signal_input, barycentric])
    signal = AngularMaxPooling()(signal)
    logits = keras.layers.Dense(output_dim)(signal)

    model = keras.Model(inputs=[signal_input, barycentric], outputs=[logits])
    return model

# Now train/validate/use it like you would with any other tensorflow model..
```

### Inputs and preprocessing

As visible in the minimal example above, the intrinsic surface convolutional layer (here geodesic convolution) expects
two inputs:
1. The signal defined on the mesh vertices (can be anything from descriptors like SHOT [5] to simple 3D-coordinates of
the vertices).
2. Barycentric coordinates for signal interpolation in the format specified by
``compute_barycentric_coordinates``.

For the latter: **geoconv** supplies you with the necessary preprocessing functions:
1. Use ``compute_gpc_systems`` on your triangle meshes (which are stored in a format that is
supported by **[Trimesh](https://trimsh.org/index.html)**, e.g. 'ply') to compute local geodesic polar coordinate
systems with the algorithm suggested by [4].
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
