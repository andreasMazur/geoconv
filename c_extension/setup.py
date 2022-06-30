from distutils.core import setup, Extension
import numpy as np


module = Extension(
    "c_extension",
    ["c_extension.c"],
    include_dirs=[np.get_include()],
    extra_link_args=["-lblas", "-lcblas"]
)


def main():
    setup(
        name="c_extension",
        version="1.0.0",
        description="A module to compute GPC.",
        ext_modules=[module]
    )


if __name__ == "__main__":
    main()
