# GeoConv

[![GitHub stars](https://img.shields.io/github/stars/andreasMazur/geoconv)](https://github.com/andreasMazur/geoconv)
[![CI](https://github.com/andreasMazur/geoconv/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/andreasMazur/geoconv/actions/workflows/python-package-conda.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/github/license/andreasMazur/geoconv)](https://github.com/andreasMazur/geoconv/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/geoconv)](https://pypi.org/project/geoconv/)

## Let's bend planes to curved surfaces.

<img align="right" style="margin-left: 10px; width: 180px;" src="geoconv_cartoon.png">

**GeoConv** is a Python library that provides end-to-end tools for deep learning on curved surfaces in 3D ambient 
spaces. That is, whether it is preprocessing your triangle meshes into a format that can be fed into neural networks, 
or the implementation of surface convolutions, GeoConv has you covered.

## Library Design and Minimal Examples

GeoConv conceptually divides into two areas: 
1. **Preprocessing**: evolves around the `Atlas`-class
2. **Surface CNN design**: evolves around the `ConvBase`-class

**Preprocessing.** GeoConv implements the preprocessing procedure of triangle meshes in 3 steps: (i) compute local 
surface charts on your triangle meshes, (ii) define template vertices, and (iii) compute barycentric 
coordinates [6]. The `Atlas`-class contains methods that implement each preprocessing step (... and more QoL features 
such as chart visualization and the possibility of applying linear chart transformations). Computing local charts, 
using any charting method of [7-10], is done by simply initializing an Atlas object:

```python
from geoconv.preprocessing.atlas import Atlas

atlas = Atlas(
    triangle_mesh=triangle_mesh,
    max_radius=max_radius,  # maximum local chart radius
    method="fmm",  # other alternatives: ["hdm", "dgpc", "tp"]
    normalization_method="hdm",
    processes=10  # number of concurrent processes used for preprocessing
)
```

The initialized object `atlas` already contains the normalized triangle mesh (geodesic diameter = 1) and all 
local surface charts. The barycentric coordinates can now be computed for any desired template configuration (i.e.,
polar grid config + template radius):

```python
atlas.determine_barycentric_coordinates(
    n_radial=n_radial,  # int
    n_angular=n_angular,  # int
    template_radius=chart_radius / 2  # float
)
```

One `Atlas` object can store multiple template configurations. Any atlas can be stored in either `hdf5`- or 
`npy`-format:

```python
atlas.save("./atlas.hdf5")  # one HDF5 file storing all information (potentially large file)
atlas.save_training_data("./atlas_dir")  # a directory containing NumPy arrays required for training (typically smaller)
```

**Surface CNN design.** While preprocessing is completely DeepLearning-library-agnostic, network design is either done
using `TensorFlow` or `Pytorch`. Currently, GeoConv provides implementations for the following surface convolutions:

- Intrinsic surface convolutions (ISCs) [1]
- Geodesic surface convolutions (GCNNs) [2]
- Harmonic surface convolutions (HSNs) [3]
- Gauge-equivariant mesh convolutions (GEM-CNNs) [4] and a radial sensitive extension (GEM-CNN+) [1]
- Equivariant mesh attention convolutions (EMANs) [5] and a radial sensitive extension (EMAN+) [1]

Any surface convolution is implemented as a subclass of the `ConvBase`-class, which contains elementary methods for
surface convolutions such as the `signal_pullback`, the `signal_pullback_with_parallel_transport` and the parametric 
`patch_operator` [11].

A minimal **TensorFlow** model could look like follows:

```python
from geoconv.tensorflow.layers import ConvGeodesic
from geoconv.tensorflow.layers import AngularMaxPooling

import tensorflow as tf


def define_model(n_vertices, feature_dim, n_radial, n_angular, template_radius, output_dim):
    """Define a geodesic convolutional neural network"""

    # Initialize input layers, one for the vertex features and another for barycentric coordinates
    signal_input = tf.keras.Input(shape=(n_vertices, feature_dim), name="image_input", dtype=tf.float32)
    barycentric = tf.keras.Input(shape=(n_vertices, n_radial, n_angular, 3, 2), name="bc_input", dtype=tf.float32)
    
    # Initialize surface convolution
    signal = ConvGeodesic(
        output_dim=32,  # return 32-dimensional output vector
        template_radius=template_radius,  # the template radius used during barycentric coordinates computation
        activation="relu",
        rotation_delta=1  # sets the skip in between of 'n_angular' applied template orientations
    )([signal_input, barycentric])
    
    # Apply max-pooling
    signal = AngularMaxPooling()(signal)
    
    # Output dense layer
    logits = tf.keras.layers.Dense(output_dim)(signal)

    # Initialize model
    model = tf.keras.Model(inputs=[signal_input, barycentric], outputs=[logits])
    return model
```

In **PyTorch**, on the other hand, one could implement the same surface CNN as follows:

```python
from geoconv.pytorch.layers import ConvGeodesic
from geoconv.pytorch.layers import AngularMaxPooling

import torch


class GCNN(torch.nn.Module):
    def __init__(self, feature_dim, n_radial, n_angular, template_radius, output_dim):
        super().__init__()
        self.geodesic_conv = ConvGeodesic(
            feature_input_dim=feature_dim,
            output_dim=32,  # return 32-dimensional output vector
            rotation_delta=1,  # sets the skip in between of 'n_angular' applied template orientations
            n_radial=n_radial,
            n_angular=n_angular,
            template_radius=template_radius,  # the template radius used during barycentric coordinates computation
            activation_fn=torch.nn.Identity(),
        )
        self.amp = AngularMaxPooling()
        self.output = torch.nn.Linear(in_features=32, out_features=output_dim)

    def forward(self, x):
        signal, barycentric = x
        signal = self.geodesic_conv([signal, barycentric])
        signal = self.amp(signal)
        return self.output(signal)
```

Eventually, the networks can be trained as any other neural network in the respective DeepLearning framework.

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

3. In case OpenGL context cannot be created:
    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```

## Intended Use, Citations and License

GeoConv provides implementations of- and interfaces to methods from publicly available prior work for computing local
coordinate systems on triangle mesh- or point cloud data and for performing surface convolutions, intended to
facilitate academic and non-commercial research on neural networks operating on surface data.

Users of GeoConv are, through its use, employing methods from prior published work and are expected to acknowledge the
original inventors of the used methods by citing the corresponding publications.

GeoConv is distributed under the terms of the **GNU General Public License v3.0 (GPL-3.0)**.

## Citation

Further information on GeoConv can be found in our paper:

```bibtex
@article{journey_through_surface_convs,
    title={A Journey Through Surface Convolutions},
    author={Andreas Mazur and David P. Leins and Fabian Hinder and Barbara Hammer},
    journal={Transactions on Machine Learning Research},
    year={2026},
    url={https://openreview.net/forum?id=lCwv0bo973}
}
```

If you are using this repository, please cite our work and the publications of the original methods you have been using.

## Referenced Literature

[1]: Andreas Mazur, David P. Leins, Fabian Hinder, and Barbara Hammer. "A Journey Through Surface Convolutions". 
     Transactions on Machine Learning Research. (2026). URL https://openreview.net/forum?id=lCwv0bo973.

[2]: Jonathan Masci, Davide Boscaini, Michael Bronstein, and Pierre Vandergheynst. Geodesic convolutional neural 
     networks on riemannian manifolds. In ICCV workshops, 2015. doi: 10.1109/ICCVW.2015.112.

[3]: Ruben Wiersma, Elmar Eisemann, and Klaus Hildebrandt. Cnns on surfaces using rotation-equivariant features. ACM 
     Trans. Graph., 2020. doi: 10.1145/3386569.3392437.

[4]: Pim De Haan, Maurice Weiler, Taco Cohen, and Max Welling. Gauge equivariant mesh cnns: Anisotropic convolutions on
     geometric graphs. In ICLR, 2021. URL https://openreview.net/forum?id=Jnspzp-oIZE.

[5]: Sourya Basu, Jose Gallego-Posada, Francesco Viganò, James Rowbottom, and Taco Cohen. Equivariant mesh attention
     networks. Transactions on Machine Learning Research, 2022. URL https://openreview.net/forum?id=3IqqJh2Ycy.

[6]: Adrien Poulenard and Maks Ovsjanikov. Multi-directional geodesic neural networks via equivariant convolution. ACM
     Trans. Graph., 2018. doi: 10.1145/3272127.3275102.

[7]: R. Kimmel and J. A. Sethian. Computing geodesic paths on manifolds. PNAS, 1998. doi: 10.1073/pnas.95.15.8431.

[8]: Eivind Lyche Melvær and Martin Reimers. Geodesic polar coordinates on polygonal meshes. In Computer Graphics Forum,
     2012. doi: 10.1111/j.1467-8659.2012.03187.x.

[9]: Samuele Salti, Federico Tombari, and Luigi Di Stefano. Shot: Unique signatures of histograms for surface and 
     texture description. Computer Vision and Image Understanding, 2014. doi: 10.1016/j.cviu.2014.04.011.

[10]: Keenan Crane, Clarisse Weischedel, and Max Wardetzky. The heat method for distance computation. Commun. ACM, 2017.
      doi: 10.1145/3131280.

[11]: Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, and Michael M. Bronstein. Geometric
      deep learning on graphs and manifolds using mixture model cnns. In CVPR, 2017. doi: 10.1109/CVPR.2017.576.
