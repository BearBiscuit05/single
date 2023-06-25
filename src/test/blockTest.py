"""
测试构建block模块满足训练要求
"""
import dgl
import numpy as np
import torch as th
from dgl.nn import SAGEConv
from dgl.heterograph import DGLBlock
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl.nn as dglnn
import mmap

def create_dgl_block(data, num_src_nodes, num_dst_nodes):
    row, col = data
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
    # g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
    g = DGLBlock(gidx)
    return g

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

if __name__ == '__main__':
    filePath = "../../../data/products/part0"
    file = open(filePath+"/feat.bin", "r+b")
    head = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_DEFAULT)
    float_size = np.dtype(float).itemsize
    nodeIDs = [i for i in range(12)]
    feats = torch.zeros((len(nodeIDs), 10), dtype=torch.float32)
    for index, nodeID in enumerate(nodeIDs):
        feat = torch.frombuffer(head, dtype=torch.float32, offset=nodeID*10*float_size, count=10)
        feats[index] = feat
    print(feats.size())
    
    # feat = torch.rand((12, 10))
    blocks = []
    s = time.time()
    dst = th.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    src = th.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11])
    block = dgl.graph((src, dst))
    block = dgl.to_block(block)
    blocks.append(block.to('cuda:0'))
    dst = th.tensor([0, 0, 0])
    src = th.tensor([1, 2, 0])
    block = dgl.graph((src, dst))
    block = dgl.to_block(block)
    blocks.append(block.to('cuda:0'))
    print(blocks)
    model = SAGE(10, 16, 3).to('cuda:0')
    # print(blocks)
    model.train()
    y_hat = model(blocks, feats.to('cuda:0'))

