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

def readGraph(rank,dataPath,datasetName):
    graph_dir = dataPath
    part_config = graph_dir + "/"+datasetName +'.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    return subg, node_feat, node_type

def gen_graph_file(data,rank,Wsize,dataPath,datasetName,savePath):
    subg, node_feat, node_type = data
    src = subg.edges()[0].tolist()
    dst = subg.edges()[1].tolist()
    src = torch.tensor(src)
    src = subg.ndata[dgl.NID][src]
    print(src)
    # inner = subg.ndata['inner_node'].tolist()
    # innernode = subg.ndata['inner_node'].sum()
    # nodeDict = {}
    # partdict = []
    # for i in range(Wsize):
    #     partdict.append({})
    # # 读取JSON文件
    # part_config = dataPath + "/"+datasetName +'.json'
    # with open(part_config, 'r') as file:
    #     SUBGconf = json.load(file)
    # # 使用读取的数据
    # boundRange = SUBGconf['node_map']['_N']
    # basiclen = SUBGconf['node_map']['_N'][rank][1] - SUBGconf['node_map']['_N'][rank][0]
    # incount = 0
    # outcount = [0 for i in range(Wsize)]
    # for index in range(len(src)):
    #     srcid,dstid = src[index],dst[index] # 
    #     if inner[srcid] == 1 and inner[dstid] == 1:
    #         if dstid not in nodeDict:
    #             nodeDict[dstid] = [dstid]
    #         nodeDict[dstid].append(srcid)
    #         incount += 1
    #     elif inner[srcid] != 1 and inner[dstid] == 1:     # 只需要dst在子图内部即可
    #         srcid = subg.ndata[dgl.NID][srcid] # srcid ：local 查询全局ID 
    #         partid = -1
    #         for pid,(left,right) in enumerate(boundRange):
    #             if left <= srcid and srcid < right:
    #                 partid = pid
    #                 break
    #         if dstid not in partdict[partid]:
    #             partdict[partid][dstid] = []
    #         # 计算合并id
    #         newsrcid = srcid - SUBGconf['node_map']['_N'][partid][0] + basiclen
    #         if srcid - SUBGconf['node_map']['_N'][partid][0] < 0:
    #             print("count error: srcid:{},partid:{},bound:{}...".format(srcid,partid,SUBGconf['node_map']['_N'][partid][0]))
    #             exit()
    #         partdict[partid][dstid].append(newsrcid)
    #         outcount[partid] += 1 

def gen_feat_file(data,rank,savePath):
    savePath = savePath + "/part"+str(rank)
    subg, node_feat, node_type = data
    nt = node_type[0]
    featInfo = node_feat[nt + '/features']
    featInfo = featInfo.detach().numpy()
    featInfo.tofile(savePath +"/feat.bin")
    print("feat-part{} processed ! ".format(rank))


if __name__ == '__main__':
    dataPath = '/home/bear/workspace/singleGNN/data/raw-products_4/'
    dataName = "ogb-product"
    savePath = '/home/bear/workspace/singleGNN/data/products_4/'
    index=4
    for rank in range(index):
        subg, node_feat, node_type = readGraph(rank,dataPath,dataName)
        data = (subg, node_feat, node_type)
        gen_graph_file(data,rank,index,dataPath,dataName,savePath)
        