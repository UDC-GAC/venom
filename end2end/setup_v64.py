from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from pybind11.setup_helpers import Pybind11Extension
import sys

setup(
    name='spatha',
    version='0.0.1',
    description='Custom library for Sparse Tensor Cores',
    author='Roberto L. Castro',
    author_email='roberto.lopez.castro@udc.es',
    ext_modules=[
            CUDAExtension('spatha',
                              ['spatha_mod/block_sparse/api/spatha.cu'],
                              extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_86', '--ptxas-options=-v', '-lineinfo', '-DV_64']})
                  ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)