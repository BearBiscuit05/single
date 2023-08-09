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
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
    #g = DGLBlock(gidx)
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
            print(h.shape)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


if __name__ == '__main__':

    feats = torch.rand((12, 10), dtype=torch.float32)
    blocks = []
    s = time.time()
    dst = th.tensor([0, 0, 0 ,1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    src = th.tensor([0, 0, 0 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    block = dgl.create_block((src , dst), num_src_nodes=12, num_dst_nodes=3)    
    print(block)

    #============[TEST]============
    src = th.tensor([i for i in range(10000)])
    dst = th.tensor([i for i in range(10000)])
    
    normalTime = time.time()
    for i in range(10000):
        block = dgl.graph((src, dst))
        block = dgl.to_block(block)
    print("normal create time:{}".format(time.time()-normalTime))

    normalTime = time.time()
    for i in range(10000):
        block = dgl.create_block((src , dst), num_src_nodes=10000, num_dst_nodes=10000)    
    print("direct create time:{}".format(time.time()-normalTime))

    normalTime = time.time()
    for i in range(10000):
        create_dgl_block((src , dst), 10000, 10000)
    print("gnnlab create time:{}".format(time.time()-normalTime))

    # data = (src,dst)
    # block = create_dgl_block(data, 12, 12)
    # print(block)

    # blocks.append(block.to('cuda:0'))

    # dst = th.tensor([0, 0, 0])
    # src = th.tensor([1, 2, 0])
    # data = (src,dst)
    # block = create_dgl_block(data, 12, 12)
    # blocks.append(block.to('cuda:0'))

    # model = SAGE(10, 16, 3).to('cuda:0')
    # model.train()
    # y_hat = model(blocks, feats.to('cuda:0'))
    # print(y_hat)
