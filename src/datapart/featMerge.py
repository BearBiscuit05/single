import numpy as np
import dgl
import torch
data = np.fromfile("/home/bear/workspace/single-gnn/src/datapart/data/partition_0.bin",dtype=np.int32)
srcs = data[::2]
dsts = data[1::2]
srcs_tensor = torch.Tensor(srcs).to(torch.int32).cuda()
dsts_tensor = torch.Tensor(dsts).to(torch.int32).cuda()
uni = torch.ones(len(dsts)*2).to(torch.int32).cuda()
sg1,sg2,sg3 = dgl.remappingNode(srcs_tensor,dsts_tensor,uni)
print("mapped src:",sg1)
print("mapped dst:",sg2)
print("mapped uni:",sg3.shape)
# filename = "/home/bear/workspace/single-gnn/data/raid/papers100M/feats.bin"
# fpr = np.memmap(filename, dtype='float32', mode='r', shape=(111059956,128))
# print(fpr)
# sg3 = sg3.cpu()
# subFeat = fpr[sg3]
# print(subFeat)