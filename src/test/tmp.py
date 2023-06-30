import numpy as np

# 假设你有一个二维数组 arr
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# 假设你有一个索引序列 t1 和一个指定修改的结果 t2
t1 = np.array([0, 2])  # 索引序列
t2 = np.array([[10, 20, 30],[3,4,5]])  # 修改的结果
print(t2.shape[0])
# 使用索引功能将索引位置的值修改为指定内容
arr[t1] = t2

# 输出修改后的数组
print(arr)
