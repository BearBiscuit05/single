#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='hashtest',
    ext_modules=[
        CUDAExtension(
            name='hashtest',
            sources=[
                'cuda_hashtable.cu',
            ],
            # include_dirs=[os.path.join(
            #     here, '3rdparty/cub')],
            # include_dirs=['./3rdparty/cub'],
            extra_link_args=['-Wl', '-fopenmp'],
            extra_compile_args={
                'cxx': ['-std=c++14', '-g',
                        '-fPIC',
                        '-Ofast',
                        '-DSXN_REVISED',
                        '-Wall', '-fopenmp', '-march=native'],
                'nvcc': ['-std=c++14',
                         '-g',
                         '-DSXN_REVISED',
                         '--compiler-options', "'-fPIC'",
                         ]
            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
