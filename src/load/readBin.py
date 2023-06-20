import time
import mmap
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

def mergeFeat(mmap_file_hand,sampleNodes,featLen):
    feats = []
    # int32_size = np.dtype(float).itemsize   
    for index,nodeID in enumerate(sampleNodes):
        int_array_length = featLen
        # offset = nodeID *featLen* int32_size
        feat_tmp = torch.frombuffer(mmap_file_hand, dtype=torch.float32, offset=nodeID*featLen, count=featLen)
        feats.append(feat_tmp)
    print(feats)
    #return feats

def gen_format_file(rank,Wsize,dataPath,datasetName,savePath):
    graph_dir = dataPath
    part_config = graph_dir + "/"+datasetName +'.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)

    node_type = node_type[0]
    featInfo = node_feat[node_type + '/features']
    #torch.save(featInfo, "feat_"+str(rank)+".bin")
    print("feat len : {}".format(len(featInfo[0])))
    print("feat: {}".format(featInfo[0:11]))
    print("data-{} processed ! ".format(rank))


if __name__ == "__main__":
    dataPath = "data"
    dataName = "ogb-product"
    savePath = "./processed_feat"
    # for i in range(4):
    #     gen_format_file(i,4,dataPath,dataName,savePath)
    gen_format_file(0,4,dataPath,dataName,savePath)


    sampleNodes = [i for i in range(11)]
    featLen = 100
    file_path = "./feat_0.bin"
    file = open(file_path, "r+b")
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    feat = mergeFeat(mmapped_file,sampleNodes,featLen)
    
    mmapped_file.close()
    file.close()