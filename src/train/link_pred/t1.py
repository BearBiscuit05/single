import torch

# 假设您有以下张量
src = torch.tensor([0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8])
dst = torch.tensor([1, 2, 3, 4, 0, 5, 7, 8, 9, 0, 6, 2, 4, 8, 0, 6, 7, 1, 3, 9, 5, 3, 5, 7, 9])
src_subset = torch.tensor([2, 0, 6])
dst_subset = torch.tensor([3, 5, 7])

# 构建子集中的边的张量
subset_edges = torch.stack((src_subset, dst_subset), dim=1)

# 找到子集边在原始张量中的索引
edge_indices = (src == subset_edges[:, 0]) & (dst == subset_edges[:, 1])
edge_indices = edge_indices.nonzero(as_tuple=False).squeeze()

print("Edge Indices:", edge_indices)
