from distutils.core import setup, Extension

import numpy as np


c_extension = Extension(
    "c_extension",
    ["./src/geoconv/preprocessing/c_extension/c_extension.c"],
    include_dirs=[np.get_include()],
    extra_link_args=["-lblas", "-lcblas"]
)

if __name__ == "__main__":
    setup(
        ext_modules=[c_extension]
    )
