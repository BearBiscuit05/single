from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name = "signn_v1",
    include_dirs = ["."],
    ext_modules = [
        CUDAExtension(
        "signn_v1",  # 扩展模块的名称
        sources = [
            "cuda_hashtable.cu",
            "cuda_mapping.cu",
            "sample_kernel.cu",
            "sample_node.cpp",
        ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension     # 执行自定义的构建操作
    },
    # packages=find_packages()
)