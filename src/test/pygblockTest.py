# from torch_geometric.datasets import Reddit
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.nn import SAGEConv

import torch
import torch as th
import copy
# from torch_geometric.data import Data

def genPYGBatchTemplate():
    zeros = torch.tensor([0])
    template_src = torch.empty(0,dtype=torch.int32)
    template_dst = torch.empty(0,dtype=torch.int32)
    ptr = batchsize + 1
    fanout = [4, 3]
    batchsize = 4
    seeds = [i for i in range(1, batchsize + 1)]
    for number in fanout:
        dst = copy.deepcopy(seeds)
        src = copy.deepcopy(seeds)
        for ids in seeds:
            for i in range(number-1):
                dst.append(ids)
                src.append(ptr)
                ptr += 1
        seeds = copy.deepcopy(src)
        template_src = torch.cat([template_src,torch.tensor(src,dtype=torch.int32)])
        template_dst = torch.cat([template_dst,torch.tensor(dst,dtype=torch.int32)])
    template_src = torch.cat([template_src,zeros])
    template_dst = torch.cat([template_dst,zeros])
    PYGTemplate = torch.stack([template_src,template_dst])
    return PYGTemplate


PYGTemplate = genPYGBatchTemplate()
features = torch.rand(PYGTemplate[0].shape[0], 10)
print(features.shape)
print(PYGTemplate)


# genPYGBatchTemplate(1,4)
# edge = torch.stack((src,dst),dim=0)

# print(template) 
data = Data(x=feats, edge_index=edge)
# print(data.edge_index)