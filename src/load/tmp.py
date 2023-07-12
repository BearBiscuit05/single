import torch

# 创建一个张量
tensor = torch.tensor([True, True, False, True, True, False])

# 将张量转换为布尔类型
bool_tensor = tensor.bool()
print(bool_tensor)
# 计算False的数量
num_false = bool_tensor.size(0) - bool_tensor.sum().item()

print("False的数量:", num_false)
