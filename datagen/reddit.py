import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
import ast
import sklearn.metrics
import numpy as np
import time


def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir='./data/dataset/')
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

g, dataset,train_idx,val_idx,test_idx= load_reddit()

src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src.tofile("/raid/bear/reddit_bin/srcList.bin")
dst.tofile("/raid/bear/reddit_bin/dstList.bin")

feat = g.ndata['feat'].numpy().tofile("/raid/bear/reddit_bin/feat.bin")
label = g.ndata['label'].numpy().tofile("/raid/bear/reddit_bin/label.bin")

torch.Tensor(train_idx).to(torch.int64).numpy().tofile("/raid/bear/reddit_bin/trainIDs.bin")
torch.Tensor(val_idx).to(torch.int64).numpy().tofile("/raid/bear/reddit_bin/valIDs.bin")
torch.Tensor(test_idx).to(torch.int64).numpy().tofile("/raid/bear/reddit_bin/testIDs.bin")