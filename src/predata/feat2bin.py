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
    featInfo = node_feat[node_type + '/features']
    featInfo = featInfo.detach().numpy()
    featInfo.tofile("feat_"+str(rank)+".bin")
    print(len(featInfo))
    print("data-{} processed ! ".format(rank))

if __name__ == '__main__':
    dataPath = "data"
    dataName = "ogb-product"
    savePath = "./processed_feat"
    # for i in range(4):
    #     gen_format_file(i,4,dataPath,dataName,savePath)
    gen_format_file(0,4,dataPath,dataName,savePath)