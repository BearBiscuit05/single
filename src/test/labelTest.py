import numpy as np
import torch
import dgl
import time

# 首先每个训练点有一个自己的标签(idx)
# 随后对标签进行传递 dst <-> src
# 在经过n轮之后,每个点都有了一个自己的最终标签
# 之后对聚类结果进行合并

# 目标  : 1.验证标签传递函数的正确性
#       : 2.将聚类结果最后聚合为指定数目
#       : 3.对标签传递函数可以流式调用(不急)    

# 读取图文件
graph = torch.as_tensor(np.fromfile("/raid/bear/data/raw/papers100M/graph1.bin",dtype=np.int32))
trainids = torch.as_tensor(np.fromfile("/raid/bear/data/raw/papers100M/trainIds.bin",dtype=np.int64))
src = graph[::2]
dst = graph[1::2]

# 图中由于是只针对训练点进行聚类，因此只对训练点附初始标签(就是本身的索引)
# 而对于其他非训练节点，标签默认是-1，在标签传递时，只会接受非负数标签，而不会主动传递
nodeNUM = 111059956
nodeLabel = torch.zeros(nodeNUM).to(torch.int32) -1
nodeLabel[trainids] = trainids.to(torch.int32)


nodeLabel = nodeLabel.cuda()
src = src.cuda()
dst = dst.cuda()
print("nodeLabel :",nodeLabel)
for _ in range(2):
    # 表示对2跳邻居进行标签传递
    dgl.lpGraph(src,dst,nodeLabel)
print("nodeLabel :",nodeLabel)

# 内存处理
src = src.cpu()
dst = dst.cpu()
nodeLabel = nodeLabel.cpu()
import gc
torch.cuda.empty_cache()
gc.collect()


# 由于只对训练点进行聚类，所以抽取训练点的标签即可
value = nodeLabel[trainids.to(torch.int64)]

# 查看聚类后的信息
binAns = torch.bincount(value)
torch.max(binAns)

# 对聚类的大小进行排序(binAns表明每个聚类有多大)
s_binAns,_ = torch.sort(binAns,descending=True)

# 测试文件
# src = torch.Tensor([0,2,4,5,3,4,2,5]).to(torch.int32).cuda()
# dst = torch.Tensor([1,3,7,6,4,2,1,3]).to(torch.int32).cuda()
# nodeLabel = torch.Tensor([-1,-1,2,3,4,-1,-1,-1]).to(torch.int32).cuda()
# print("nodeLabel :",nodeLabel)
# dgl.lpGraph(src,dst,nodeLabel)
# print("nodeLabel :",nodeLabel)