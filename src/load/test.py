import time
import copy
import torch
import dgl

dst = [0,0,0,1,1,1,2,2,2]
src = [4,5,6,7,8,9,10,11,12]
dst = torch.Tensor(dst).to(torch.int64).to('cuda:0')
src = torch.Tensor(src).to(torch.int64).to('cuda:0')
print(src)
print(dst)
all = torch.cat([src,dst])
print(all)
