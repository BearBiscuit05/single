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

def load_partition(rank,nodeID):
    graph_dir = 'data_8/'
    part_config = graph_dir + 'ogb-product.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    ids = subg.ndata[dgl.NID].tolist()
    inner = subg.ndata['inner_node'].tolist()
    print(subg.nodes())
    print(subg.edges())

def read_subG(rank):
    graph_dir = 'data_4/'
    part_config = graph_dir + 'ogb-product.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    in_graph = dgl.node_subgraph(subg, subg.ndata['inner_node'].bool())
    in_nodes = torch.arange(in_graph.num_nodes())
    out_graph = subg.clone()
    out_graph.remove_edges(out_graph.out_edges(in_nodes, form='eid'))
    """
        node_feat: only inner node
        -- _N/features
        -- _N/labels
        -- _N/train_mask
        -- _N/val_mask
        -- _N/test_mask
    """
    print(f'Process {rank} has {subg.num_nodes()} nodes, {subg.num_edges()} edges ')
          #f'{in_graph.num_nodes()} inner nodes, and {in_graph.num_edges()} inner edges.')
    print("nodeNUM:",subg.num_nodes())
    print("featLen:",len(node_feat['_N/features']))
    print("NID    :",subg.ndata[dgl.NID])
    print("nodes():",subg.nodes()) # 本地子图序列
    print("gpb    :",gpb.partid2nids(rank))
    print("edge   :",subg.edges())  
    # print(len(node_feat['_N/labels']))

def save_coo_bin(nodeDict, filepath, nodeNUM, edgeNUM, basicSpace):
    srcList = []
    range_list = []
    place = 0
    print(nodeNUM)
    for key in range(nodeNUM):
        if key in nodeDict:
            srcs = nodeDict[key]
            srcs.sort()  
            nodeLen = len(srcs)
            space = max(math.ceil(nodeLen * 1.2),basicSpace)
            extended_srcs = srcs + [0] * (space - nodeLen)
            srcList.extend(extended_srcs)
            range_list.extend([place,place+nodeLen])
            place += space
        else:
            srcList.append(key)
            range_list.extend([place,place+1])
            place += 1

    # 存储
    srcList = np.array(srcList,dtype=np.int32)
    range_list = np.array(range_list,dtype=np.int32)
    srcList.tofile(filepath+"/srcList.bin")
    range_list.tofile(filepath+"/range.bin")

def save_edges_bin(nodeDict, filepath, haloID, nodeNUM, edgeNUM):
    edges = []
    for key in range(nodeNUM):
        if key in nodeDict:
            srcs = nodeDict[key]
            for srcid in srcs:
                edges.extend([srcid,key])
    edges = np.array(edges,dtype=np.int32)
    edges.tofile(filepath+"/halo"+str(haloID)+".bin")

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
    inner = subg.ndata['inner_node'].tolist()
    innernode = subg.ndata['inner_node'].sum()
    nodeDict = {}
    partdict = []
    for i in range(Wsize):
        partdict.append({})
    # 读取JSON文件
    part_config = dataPath + "/"+datasetName +'.json'
    with open(part_config, 'r') as file:
        SUBGconf = json.load(file)
    # 使用读取的数据
    boundRange = SUBGconf['node_map']['_N']
    basiclen = SUBGconf['node_map']['_N'][rank][1] - SUBGconf['node_map']['_N'][rank][0]
    incount = 0
    outcount = [0 for i in range(Wsize)]
    for index in range(len(src)):
        srcid,dstid = src[index],dst[index] # 
        if inner[srcid] == 1 and inner[dstid] == 1:
            if dstid not in nodeDict:
                nodeDict[dstid] = [dstid]
            nodeDict[dstid].append(srcid)
            incount += 1
        elif inner[srcid] != 1 and inner[dstid] == 1:     # 只需要dst在子图内部即可
            srcid = subg.ndata[dgl.NID][srcid] # srcid ：local 查询全局ID 
            partid = -1
            for pid,(left,right) in enumerate(boundRange):
                if left <= srcid and srcid < right:
                    partid = pid
                    break
            if dstid not in partdict[partid]:
                partdict[partid][dstid] = []
            # 计算合并id
            newsrcid = srcid - SUBGconf['node_map']['_N'][partid][0] + basiclen
            if srcid - SUBGconf['node_map']['_N'][partid][0] < 0:
                print("count error: srcid:{},partid:{},bound:{}...".format(srcid,partid,SUBGconf['node_map']['_N'][partid][0]))
                exit()
            partdict[partid][dstid].append(newsrcid)
            outcount[partid] += 1 
    
    save_coo_bin(nodeDict,savePath+"/part"+str(rank),boundRange[rank][1] - boundRange[rank][0], incount,20)
    for i in range(Wsize):
        save_edges_bin(partdict[i], savePath+"/part"+str(rank), i, boundRange[rank][1] - boundRange[rank][0], outcount[i])

    print("graphdata-part{} processed ! ".format(rank))

def gen_labels_file(data,rank,savePath):
    savePath = savePath + "/part"+str(rank)
    subg, node_feat, node_type = data
    nt = node_type[0]
    labelInfo = node_feat[nt + '/labels']
    labelInfo = labelInfo.to(torch.int32).detach().numpy()
    labelInfo.tofile(savePath + "/label.bin")
    print("label-part{} processed ! ".format(rank))

def gen_ids_file(data,rank,savePath):
    savePath = savePath + "/part"+str(rank)
    subg, node_feat, node_type = data
    nt = node_type[0]
    train_mask = node_feat[nt + '/train_mask']
    val_mask = node_feat[node_type + '/val_mask']
    test_mask = node_feat[node_type + '/test_mask']
    torch.save(train_mask, savePath + "/trainID.bin")
    torch.save(train_mask, savePath + "/valID.bin")
    torch.save(train_mask, savePath + "/testID.bin")
    print("ids-part{} processed ! ".format(rank))

def gen_feat_file(data,rank,savePath):
    savePath = savePath + "/part"+str(rank)
    subg, node_feat, node_type = data
    nt = node_type[0]
    featInfo = node_feat[nt + '/features']
    featInfo = featInfo.detach().numpy()
    featInfo.tofile(savePath +"/feat.bin")
    print("feat-part{} processed ! ".format(rank))


if __name__ == '__main__':
    dataPath = "./../../data/raw-reddit_8"
    dataName = "reddit"
    savePath = "./../../data/reddit_8"
    index=8
    for rank in range(index):
        subg, node_feat, node_type = readGraph(rank,dataPath,dataName)
        data = (subg, node_feat, node_type)
        gen_graph_file(data,rank,index,dataPath,dataName,savePath)
        gen_labels_file(data,rank,savePath)
        gen_feat_file(data,rank,savePath)
        gen_ids_file(data,rank,savePath)
        