from setuptools import setup
import numpy as np
from Cython.Build import cythonize


setup(
    name="convolve",
    ext_modules=cythonize("zx.pyx"),
    zip_safe=False,
    include_dirs=[np.get_include()],
)
