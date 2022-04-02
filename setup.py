from setuptools import setup, find_packages
import numpy as np
from distutils.extension import Extension
from Cython.Build import cythonize


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'secret.metrics.rank_cylib.rank_cy',
        ['secret/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]


setup(name='ReIDSecret',
      version='1.0.0',
      packages=find_packages(),
      ext_modules=cythonize(ext_modules))
