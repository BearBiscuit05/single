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


def gen_format_file(rank,Wsize,dataPath,datasetName,savePath):
    graph_dir = dataPath
    part_config = graph_dir + "/"+datasetName +'.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    train_mask = node_feat[node_type + '/train_mask']
    # torch.save(train_mask, "trainID_"+str(rank)+".bin")
    print(train_mask)
    print("data-{} processed ! ".format(rank))

if __name__ == '__main__':
    dataPath = "data"
    dataName = "ogb-product"
    savePath = "./processed_feat"
    # gen_format_file(0,4,dataPath,dataName,savePath)
    for i in range(4):
        gen_format_file(i,4,dataPath,dataName,savePath)