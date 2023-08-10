from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name = "signn",
    include_dirs = ["test_include"],
    ext_modules = [
        CUDAExtension(
        "signn",  # 扩展模块的名称
        sources = ["test_torch/sample_node.cpp",
            "test_kernel/sample_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension     # 执行自定义的构建操作
    },
    # packages=find_packages()
)