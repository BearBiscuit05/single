# 添加函数守则

## setup.py

正常情况下不需要修改，如果有额外文件添加，则进行修改

## signn.h

cuda内部构建需要的函数声明

## sample_node.cpp

用于与torch进行交互，并且绑定pybind11

sample_node.cpp与signn.h需要同时修改

## xx.cu

构建cuda内部函数的具体实现