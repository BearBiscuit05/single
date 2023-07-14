"""
测试特征转换不产生错误
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
import mmap

def gen_format_file(rank,Wsize,dataPath,datasetName,savePath):
    graph_dir = dataPath
    part_config = graph_dir + "/"+datasetName +'.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    featInfo = node_feat[node_type + '/features']
    np_featInfo = featInfo.detach().numpy()
    #featInfo.tofile(graph_dir+"/../products/part"+str(rank)+"/feat.bin")
    # print(featInfo[:10])
    # print("data-{} processed ! ".format(rank))
    return featInfo[:10]

def mmap_read(head,rank,featLen):
    float_size = np.dtype(np.float32).itemsize
    nodeIDs = [i for i in range(10)]
    feats = torch.zeros((len(nodeIDs), featLen), dtype=torch.float32)
    for index, nodeID in enumerate(nodeIDs):
        feat = torch.frombuffer(head, dtype=torch.float32, offset=nodeID*featLen*float_size, count=featLen)
        feats[index] = feat
    return feats

if __name__ == '__main__':
    dataPath = "./../../data/raw-products"
    dataName = "ogb-product"
    savePath = "./processed_feat"

    filePath = "../../data/products/part"+str(0)
    file = open(filePath+"/feat.bin", "r+b")
    head = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_DEFAULT)
    t1 = gen_format_file(0,4,dataPath,dataName,savePath)
    t2 = mmap_read(head,0,100)
    print(t2)
    ans = torch.eq(t1,t2)
    ans = torch.all(ans)
    print(ans)