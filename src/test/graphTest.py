"""
测试数据转换过程中的数据不出现错误
"""
import os
import scipy
import dgl
from dgl.data import RedditDataset, YelpDataset
from dgl.distributed import partition_graph
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
import csv
import torch
import json
import pickle
import struct
import sys
import math
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
def NeighborTest(nodeID):
    # 测试节点
    # 比较两个List
    pass

def trans2GID():
    pass

def loadingDGLdata(datapath,rank,seeds):
    # DGL切割后的测试
    graph_dir = datapath
    part_config = graph_dir + '/ogb-product.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    sampler = MultiLayerFullNeighborSampler(1)
    dataloader = DataLoader(subg, torch.arange(seeds), sampler)
    inner = subg.ndata['inner_node']
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        #maximum = torch.max(input_nodes)
        print(input_nodes)

def loadingProcessedData(datapath,rank,wsize,number):
    # 处理后数据
    datapath += "/part"+str(rank)
    srcdata = np.fromfile(datapath+"/srcList.bin", dtype=np.int32)
    rangedata = np.fromfile(datapath+"/range.bin", dtype=np.int32)
    halo = []
    for i in range(wsize):
        data = np.fromfile(datapath+"/halo"+str(i)+".bin", dtype=np.int32)
        halo.append(data)

    seeds = [i for i in range(number)]
    data = []
    for i in seeds:
        print(srcdata[rangedata[i*2]:rangedata[i*2+1]])
    print(halo)
if __name__ == '__main__':
    processed = loadingProcessedData("./../../data/products",0,4,5)



