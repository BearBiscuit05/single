import numpy as np
import torch
import dgl
import time

# First each training node has its own label (idx)
# The tag is then passed dst <-> src
# After n rounds, each node has its own final label
# The clustering results are then merged

# Goal: 1. Verify the correctness of the label transfer function
# : 2. Aggregate the clustering results to the specified number
# : 3. The label transfer function can be called streaming (not urgent)

# read graph file
graph = torch.as_tensor(np.fromfile("/raid/bear/data/raw/papers100M/graph1.bin",dtype=np.int32))
trainids = torch.as_tensor(np.fromfile("/raid/bear/data/raw/papers100M/trainIds.bin",dtype=np.int64))
src = graph[::2]
dst = graph[1::2]

# Since only training points are clustered in the figure, 
# only initial labels (which are their own indexes) are attached to training points.
# For other non-trained nodes, the label defaults to -1,
# and only non-negative labels are accepted when the label is passed, rather than actively passed
nodeNUM = 111059956
nodeLabel = torch.zeros(nodeNUM).to(torch.int32) -1
nodeLabel[trainids] = trainids.to(torch.int32)


nodeLabel = nodeLabel.cuda()
src = src.cuda()
dst = dst.cuda()
print("nodeLabel :",nodeLabel)
for _ in range(2):
    # Indicates that the label is passed to the 2-hop neighbor
    dgl.lpGraph(src,dst,nodeLabel)
print("nodeLabel :",nodeLabel)

# Memory processing
src = src.cpu()
dst = dst.cpu()
nodeLabel = nodeLabel.cpu()
import gc
torch.cuda.empty_cache()
gc.collect()


# Since only the training points are clustered, 
# the labels of the training points can be extracted
value = nodeLabel[trainids.to(torch.int64)]

# View the information after clustering
binAns = torch.bincount(value)
torch.max(binAns)

# Order the size of the clusters (binAns indicates how big each cluster is)
s_binAns,_ = torch.sort(binAns,descending=True)

# test files
# src = torch.Tensor([0,2,4,5,3,4,2,5]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,1,3]).to(torch.int32).cuda()
# nodeLabel = torch.Tensor([-1,-1,2,3,4,-1,-1,-1]).to(torch.int32).cuda()
# print("nodeLabel :",nodeLabel)
# dgl.lpGraph(src,dst,nodeLabel)
# print("nodeLabel :",nodeLabel)