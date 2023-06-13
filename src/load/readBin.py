import numpy as np
import torch


# 使用NumPy读取二进制文件
data = np.fromfile('file.bin', dtype=np.int32)

# 将NumPy数组转换为PyTorch张量
tensor_data = torch.from_numpy(data)

# 打印张量
print(tensor_data)
