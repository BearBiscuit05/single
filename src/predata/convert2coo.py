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

def gen_subG_csv(rank):
    graph_dir = 'data_4/'
    part_config = graph_dir + 'reddit.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    inner = subg.ndata['inner_node'].tolist()
    src = subg.edges()[0].tolist()
    dst = subg.edges()[1].tolist()
    nodeID =0
    edgeNUM = 0
    with open("./edge.csv", 'a+') as f:
        csv_writer = csv.writer(f)
        for i in range(len(src)):
            if inner[src[i]] == 1 and inner[dst[i]] == 1 : 
                if src[i] > nodeID:
                    nodeID = src[i]
                if dst[i] > nodeID:
                    nodeID = dst[i]
                edgeNUM += 1
                csv_writer.writerow([src[i],dst[i]])
                
    with open("./conf.csv", 'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([nodeID,edgeNUM])

def read_subG(rank):
    graph_dir = 'data_4/'
    part_config = graph_dir + 'ogb-product.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    #inner = subg.ndata['inner_node'].tolist()
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

def write_format(nodeDict,fileName, nodeNUM, edgeNUM):
    with open(fileName, 'wb') as file:
        file.write(struct.pack('i', nodeNUM))
        file.write(struct.pack('i', edgeNUM))
        for key, values in nodeDict.items():
            file.write(struct.pack('i', -key))  # 将负数形式的key写入文件
            for value in values:
                file.write(pickle.dumps(value))  # 将value序列化后写入文件

def save_dict_to_txt(nodeDict, filename, nodeNUM, edgeNUM):
    with open(filename, 'w') as file:
        file.write(f"{nodeNUM},{edgeNUM}")
        file.write("\n")
        for key, values in nodeDict.items():
            file.write(f"-{key}")
            for value in values:
                file.write(f",{value}")
            file.write("\n")

def gen_format_file(rank,Wsize):
    """ 
    非压缩：二进制存储
        subG:只包含本位
            -src,id1,id2,id3
        bound:包含边界
            PART1
                -src,id1,id2,id3
            PART2
                -src,id1,id2,id3
        feat:
            [feat1]
            [feat2]
    """
    graph_dir = '../algo/data_4/'
    part_config = graph_dir + 'ogb-product.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)

    src = subg.edges()[0].tolist()
    dst = subg.edges()[1].tolist()
    inner = subg.ndata['inner_node'].tolist()
    innernode = subg.ndata['inner_node'].sum()
    nodeDict = {}
    partdict = []
    for i in range(Wsize):
        partdict.append({})
    # 读取JSON文件
    with open(part_config, 'r') as file:
        SUBGconf = json.load(file)
    # 使用读取的数据
    boundRange = SUBGconf['node_map']['_N']
    incount = 0
    outcount = [0 for i in range(Wsize)]
    for index in range(len(src)):
        srcid,dstid = src[index],dst[index]
        if inner[srcid] == 1 and inner[dstid] == 1:
            if srcid not in nodeDict:
                nodeDict[srcid] = []
            nodeDict[srcid].append(dstid)
            incount += 1
        elif inner[srcid] != 1:     # 只需要dst在子图内部即可
            srcid = subg.ndata[dgl.NID][srcid]
            partid = int((srcid / innernode))
            if partid > Wsize:
                partid = Wsize - 1
            if partid > 0 and partid <= Wsize - 1 and boundRange[partid][0] <= srcid and boundRange[partid][1] > srcid:
                pass
            elif partid > 0 and boundRange[partid-1][0] <= srcid and boundRange[partid-1][1] > srcid:
                partid -= 1
            elif partid < Wsize - 1 and boundRange[partid+1][0] <= srcid and boundRange[partid+1][1] > srcid:
                partid += 1
            else:
                print("src error id: ",srcid)
                exit(-1)
            if dstid not in partdict[partid]:
                partdict[partid][dstid] = []
            partdict[partid][dstid].append(srcid)
            outcount[partid] += 1         
        # elif inner[dstid] != 1:
        #     dstid = subg.ndata[dgl.NID][dstid]
        #     partid = int((dstid / innernode))
        #     if partid > Wsize:
        #         partid = Wsize - 1
        #     if partid > 0 and partid <= Wsize - 1 and boundRange[partid][0] <= dstid and boundRange[partid][1] > dstid:
        #         pass
        #     elif partid > 0 and boundRange[partid-1][0] <= dstid and boundRange[partid-1][1] > dstid:
        #         partid -= 1
        #     elif partid < Wsize - 1 and boundRange[partid+1][0] <= dstid and boundRange[partid+1][1] > dstid:
        #         partid += 1
        #     else:
        #         print("dst error id: ",dstid)
        #         exit(-1)
        #     if srcid not in partdict[partid]:
        #         partdict[partid][srcid] = []
        #     partdict[partid][srcid].append(dstid)
        #     outcount[partid] += 1
    save_dict_to_txt(nodeDict,'../data/subg_'+str(rank)+'.txt', len(nodeDict), incount)
    for i in range(Wsize):
        save_dict_to_txt(partdict[i],'../data/subg_'+str(rank)+'_bound_'+str(i)+'.txt', len(partdict[i]), outcount[i])



nodeID = 1
# PART = 16
# for i in range(PART):
#     nodeID = load_partition(i,nodeID)
gen_format_file(0,4)