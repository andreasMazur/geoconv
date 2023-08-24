# GeoConv

**Geo**desic **conv**olutional neural networks belong to the category of intrinsic mesh CNNs [1].
They operate directly on object surfaces, therefore expanding the application of convolutions
to non-Euclidean data. The **GeoConv** library delivers an implementation of the **geodesic convolution** [2] 
while taking into account suggestions for implementation details which were given in [3].
Additionally, all required preprocessing functions, like computing geodesic polar coordinates on triangulated object
meshes [4] and the computation of Barycentric coordinates, are included too.

## Note

This repository is still in development. It was tested using **Python 3.10.11**.

## Installation
1. Install **[BLAS](https://netlib.org/blas/#_reference_blas_version_3_10_0)** and **[CBLAS](https://netlib.org/blas/#_cblas)**:
     ```bash
     sudo apt install libatlas-base-dev
     ```
2. Clone and install **geoconv**:
     ```bash
     git clone https://github.com/andreasMazur/geoconv.git
     cd ./geoconv
     pip install -r requirements.txt
     pip install .
     ```
   
3. If one wishes to use a GPU (compare https://www.tensorflow.org/install/pip):
   ```bash
   conda install -c conda-forge cudatoolkit=11.8.0
   python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   ```
4. In case `libdevice` cannot be found:
    ```bash
    echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(dirname $(find / -type d -name nvvm 2>/dev/null))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

### Hint:

In case you cannot install **BLAS** or **CBLAS** you have the option to remove the ``ext_modules``-argument within
``setup.py``. This will prohibit you from taking advantage of the c-extension module that accelerates GPC-system
computation. This, in turn, implies that you have to set the ``use_c``-flag in the respective functions to ``False`` such that
the Python-alternative implementation is used.

## Usage

In order to make the usage as simple as possible, we implement the geodesic convolution as a Tensorflow-layer.
Thus, it can be used as any other default Tensorflow layer, making it easy to the users who are familiar 
with Tensorflow.

### Define a GCNN with geoconv:

```python
from geoconv.layers.conv_geodesic import ConvGeodesic
from geoconv.layers.angular_max_pooling import AngularMaxPooling
from tensorflow import keras
import tensorflow as tf


def define_model(signal_shape, barycentric_shape, output_dim):
    """Define a geodesic convolutional neural network"""

    signal_input = keras.layers.InputLayer(shape=signal_shape)
    barycentric = keras.layers.InputLayer(shape=barycentric_shape)
    signal = ConvGeodesic(
         output_dim=32,
         amt_kernel=2,
         kernel_radius=0.028
    )([signal_input, barycentric])
    signal = AngularMaxPooling()(signal)
    logits = keras.layers.Dense(output_dim)(signal)

    model = tf.keras.Model(inputs=[signal_input, barycentric], outputs=[logits])
    return model

# Now train/validate/use it like you would with any other tensorflow model..
```

### Inputs and preprocessing

As visible in the minimal example above, the geodesic convolutional layer expects two inputs:
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

For a more thorough explanation please take a look on the demo file ``preprocess_demo`` in the examples
folder.

## Cite

Using my work? Please cite this repository by using the **"Cite this repository"-option** of GitHub
in the right panel.

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
