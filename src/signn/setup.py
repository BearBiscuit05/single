from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name = "signn_v1",
    include_dirs = ["."],
    ext_modules = [
        CUDAExtension(
        "signn_v1", 
        sources = [
            "cuda_hashtable.cu",
            "cuda_mapping.cu",
            "sample_kernel.cu",
            "sample_node.cpp",
        ],
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
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension 
    },
)