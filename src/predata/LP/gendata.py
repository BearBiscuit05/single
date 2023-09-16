import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import time
import sys
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
import random

# 加载图数据结构
# 构建链路预测数据集
"""
选择训练点   --> 选择正边(1)  --> 选择负边(1)  --> 构建负边(500)
"""

def custom_sampler(g, num_samples, num_neg_samples):
    # 与原图比较，src相同，但是dst不同,如果有多个负采样,则都保持在一起
    sampled_edge_ids = random.sample(range(g.num_edges()), num_samples)
    raw_dst=torch.Tensor(g.edges()[1][sampled_edge_ids])
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(num_neg_samples)
    src, negdst = neg_sampler(g, torch.Tensor(sampled_edge_ids).to(torch.int64))
    return src, raw_dst ,negdst

def genTrainNodes():
    pass

def genTestNodes():
    pass

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir='/home/bear/workspace/singleGNN/data/dataset/')
    g = data[0]
    g.ndata['feat'] = g.ndata.pop('feat')
    g.ndata['label'] = g.ndata.pop('label')
    train_idx = []
    val_idx = []
    test_idx = []
    for index in range(len(g.ndata['train_mask'])):
        if g.ndata['train_mask'][index] == 1:
            train_idx.append(index)
    for index in range(len(g.ndata['val_mask'])):
        if g.ndata['val_mask'][index] == 1:
            val_idx.append(index)
    for index in range(len(g.ndata['test_mask'])):
        if g.ndata['test_mask'][index] == 1:
            test_idx.append(index)
    return g, data,train_idx,val_idx,test_idx

def loadingGraph(datasetName):
    if datasetName == 'ogb-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="/home/bear/workspace/singleGNN/data/dataset"))
        g = dataset[0]
    elif datasetName == 'Reddit':
        g, dataset,train_idx,val_idx,test_idx= load_reddit()
    elif datasetName == 'ogb-papers100M':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M',root="/home/bear/workspace/singleGNN/data/dataset"))
        g = dataset[0]
    return g

if __name__ == '__main__':
    # 加载数据dgl图

    # 构建训练集/测试集(正边,负边)
    pass

    # 对结果进行存储
