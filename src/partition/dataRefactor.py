import torch
import dgl
import numpy as np
import mmap

#data = np.arange(1000).reshape(100, 10)

array_file = "./part2bin/numpy_array_data.bin"

#data.tofile(array_file)

fpr = np.memmap(array_file, dtype='int64', mode='r', shape=(100,10))
indices = [1, 3, 5, 7]
get = fpr[indices]
print(get)

# print("Array data saved to:", array_file)

# 打开二进制文件，并使用 mmap 访问数据
# with open(array_file, "rb") as f:
#     with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_data:
#         # 根据需要的索引读取数据
#         indices = [1, 3, 5, 7]
#         selected_rows = [np.frombuffer(mmapped_data[idx * data.itemsize * data.shape[1]:
#                                                       (idx + 1) * data.itemsize * data.shape[1]],
#                                        dtype=data.dtype) for idx in indices]

# # 将选定的行存储到一个新的数组中
# selected_data_array = np.stack(selected_rows)

# print("Selected data array:\n", selected_data_array)

# # 将选定的数据存储到另一个文件中
# selected_data_file = "./part2bin/selected_data.npy"
# np.save(selected_data_file, selected_data_array)

# print("Selected data saved to:", selected_data_file)
# # 重新读取保存的选定数据数组
# loaded_selected_data = np.load(selected_data_file)

# print("Loaded selected data array:\n", loaded_selected_data)

# 构建子图内部特征

"""
--part0
----inter edges
----cut edges  --> 需要按照另外一个分区来对应id
----inter nodes
--part1
...
--result:记录了每个分区的节点数目

转换过程:
1.part-n来说,global id被修改为part-(n-1)范围的最大--part-n的范围
2.此时需要存储一个映射表-->mapping文件
3.修改cut edges中一边的id内容 --> lid
4.按照mapping对应并修改到另外一个id(要匹配到) --> nodeNUM + lid 
"""
def fetchFeat(featFilePath,nodeNUM,FeatLen,indices):
    fpr = np.memmap(featFilePath, dtype='float64', mode='r', shape=(nodeNUM,FeatLen))
    feats = fpr[indices]
    return feats
     
def fetchLabel(labelFilePath,indices):
    label_data = np.fromfile(labelFilePath, dtype=np.int64)
    label_selected_data = label_data[indices]
    return label_selected_data

def fetchNodeClass():
    # 需要从mask中进行提取
    pass


if __name__ == '__main__':
    pass
