"""
测试图数据转换为二进制文件时不产生错误
"""
import numpy as np

file1_data = np.fromfile("./../../data/products/part0/halo1.bin", dtype=np.int32)
file2_data = np.fromfile("./../../data/products/part0/halo2.bin", dtype=np.int32)
file3_data = np.fromfile("./../../data/products/part0/tmp_halo1.bin", dtype=np.int32)
file4_data = np.fromfile("./../../data/products/part0/tmp_halo2.bin", dtype=np.int32)
# is_equal = np.array_equal(file1_data_trimmed, file2_data)
# print("The data is equal:", is_equal)


print(file1_data[:10])
print(file2_data[:10])
print(file3_data[:10])
print(file4_data[:10])