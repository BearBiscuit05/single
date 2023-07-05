from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

import torch
import copy
from torch_geometric.data import Data


def genBlockTemplate():
    template = []
    blocks = []
    ptr = 0
    fanout=[4,4]
    batchsize=4
    seeds = [i for i in range(1,batchsize+1)]
    for number in fanout:
        dst = copy.deepcopy(seeds)
        src = copy.deepcopy(seeds)
        ptr = len(src) + 1    
        for ids in seeds:
            for i in range(number-1):
                dst.append(ids)
                src.append(ptr)
                ptr += 1
        seeds = copy.deepcopy(src)
        src.append(0)
        dst.append(0)
        template.insert(0,[torch.tensor(src),torch.tensor(dst)])
    return template

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# template = genBlockTemplate()
edge = torch.stack((template[-1][0],template[-1][1]),dim=0)
# print(edge)
# data = Data(x=x, edge_index=edge)
# print(data.edge_index)
