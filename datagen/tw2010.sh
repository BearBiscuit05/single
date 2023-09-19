#!/bin/bash

# 下载源数据

# 生成特征数据集文件
dd if=/dev/zero of=output.bin bs=4 count=10

# 生成训练集/验证集/测试集
