from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name = "self_sample_full",
    include_dirs = ["include"],
    ext_modules = [
        CUDAExtension(
        "sample_hop",  # 扩展模块的名称
        sources = ["torch/sample_node_ops.cpp", "kernel/sample_node_kernel.cu"],
        # extra_compile_args = {
        #         'cxx': ['-DDEFAULT_ARG_VALUE=0'],  # 设置默认参数的值
        #         'nvcc': ['-DDEFAULT_ARG_VALUE=0'],  # 设置默认参数的值
        #     },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension     # 执行自定义的构建操作
    },
    # packages=find_packages()
)