from distutils.core import setup, Extension
from setuptools import find_packages

import numpy as np

c_extension = Extension(
    "c_extension",
    ["./geoconv/preprocessing/c_extension/c_extension.c"],
    include_dirs=[np.get_include()],
    extra_link_args=["-lblas", "-lcblas"]
)

if __name__ == "__main__":

    setup(
        author="Andreas Mazur",
        name="geoconv",
        version="1.0.0",
        packages=find_packages(),
        license="GNU General Public License v3.0",
        description="An implementation for the geodesic convolution.",
        long_description=open("README.md").read(),
        url="https://github.com/andreasMazur/GeodesicConvolution",
        ext_modules=[c_extension]
    )
