import torch
import time

# 创建一个包含 15 万个随机整数的向量
num_nodes = 250000
random_data = torch.randint(low=0, high=1000000, size=(num_nodes,), dtype=torch.int32)

# 测试开始时间
start_time = time.time()

# 使用 torch.max 函数找到最大值和对应索引
max_value, max_index = torch.max(random_data, dim=0)

# 测试结束时间
end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time

print(f"Max value: {max_value.item()}")
print(f"Index of max value: {max_index.item()}")
print(f"Elapsed time: {elapsed_time:.5f} seconds")
