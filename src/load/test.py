import numpy as np

# 读取 npz 文件
data = np.load('../../data/dataset/ogbn_papers100M/raw/data.npz')

# 打印 npz 文件中的数组名
print("Arrays in the npz file:", data.files)

# 获取特定数组的值
array_name = 'edge_index'  # 替换为实际的数组名
array_value = data[array_name]

# 打印数组的值
print(array_value)
