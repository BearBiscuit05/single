import numpy as np

# 定义图的节点数量和边的数量
num_nodes = 10  # 0-32 共33个节点
num_edges = 25  # 需要的边的数量

# 创建所有可能边的列表
all_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes)]
id = [i for i in range(2)]
# 随机选择40条边
selected_edges = np.random.choice(len(all_edges), num_edges, replace=False)

# 创建src和dst数组
src = np.array([all_edges[i][0] for i in selected_edges])
dst = np.array([all_edges[i][1] for i in selected_edges])
id = np.array(id)
src.tofile("/raid/bear/test_dataset/srcList.bin")
dst.tofile("/raid/bear/test_dataset/dstList.bin")
id.tofile("/raid/bear/test_dataset/trainIDs.bin")
# 打印生成的数组
print("id:",id)
print("src:", src)
print("dst:", dst)
