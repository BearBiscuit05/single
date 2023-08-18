import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import signn
import time
import struct
import os
import copy


def haloTest(graphEdge,boundList):
    file_path = './../../../data/products_4/part0/srcList.bin'
    graphEdge1 = np.fromfile(file_path, dtype=np.int32)
    graphEdge1 = torch.tensor(graphEdge1).to('cuda:0')
    file_path = "./../../../data/products_4/part0/range.bin"
    boundList1 = np.fromfile(file_path, dtype=np.int32)
    boundList1 = torch.tensor(boundList1).to('cuda:0')
    nodeNUM = int((len(boundList1))/2)
    edgeLen = len(graphEdge1)

    file_path = './../../../data/products_4/part3/srcList.bin'
    graphEdge2 = np.fromfile(file_path, dtype=np.int32)
    graphEdge2 = torch.tensor(graphEdge2).to('cuda:0')
    file_path = "./../../../data/products_4/part3/range.bin"
    boundList2 = np.fromfile(file_path, dtype=np.int32)
    boundList2 = torch.tensor(boundList2).to('cuda:0')
    graphEdge2 = graphEdge2 + int(nodeNUM)
    boundList2 = boundList2 + int(edgeLen)

    graphEdge = torch.cat([graphEdge1,graphEdge2])
    boundList = torch.cat([boundList1,boundList2])

    file_path = './../../../data/products_4/part0/halo3.bin'
    halo2 = np.fromfile(file_path, dtype=np.int32)
    halo2 = torch.tensor(halo2).to('cuda:0')
    file_path = "./../../../data/products_4/part0/halo3_bound.bin"
    halobound = np.fromfile(file_path, dtype=np.int32)
    halobound = torch.tensor(halobound).to('cuda:0')
    # print("halo nodeNUM:",len(halobound)-1)
    graphEdge_tmp = copy.deepcopy(graphEdge).to('cpu')
    start = time.time()
    signn.torch_graph_halo_merge(graphEdge,boundList,halo2,halobound,nodeNUM)
    #print("comput time:",time.time()-start)
    # graphEdge = graphEdge.to('cpu')
    # boundList = boundList.to('cpu')
    #are_equal = torch.equal(graphEdge, graphEdge_tmp)
    
    
    return graphEdge,boundList


def right_Test(graphEdge,boundList):
    # graphEdge = [i for i in range(30)]
    # boundList = [0,8,8,15,17,23,24,30]
    #graphEdge = torch.tensor(graphEdge).to(torch.int).to('cuda:0')
    #boundList = torch.tensor(boundList).to(torch.int).to('cuda:0')
    # print(graphEdge.device)
    # print(boundList.device)
    seed_num = 10
    seed = [i for i in range(seed_num)]
    seed = torch.Tensor(seed).to(torch.int).to('cuda:0')
    
    fanout = 5
    out_src = [-1 for i in range(seed_num*fanout)]
    out_src = torch.Tensor(out_src).to(torch.int).to('cuda:0')
    out_dst = [-1 for i in range(seed_num*fanout)]
    out_dst = torch.Tensor(out_dst).to(torch.int).to('cuda:0')
    start = time.time()
    out_num = torch.Tensor([0]).to(torch.int64).to('cuda:0')
    signn.torch_sample_hop(graphEdge,boundList,seed,seed_num,fanout,out_src,out_dst,out_num)
    print("comput time:",time.time()-start)
    print("out_dst :",out_dst)
    print("out_num : ",out_num)
    count = torch.sum(out_dst == -1).item()
    print("count :",seed_num*fanout - count)

    all_node = torch.cat([out_dst[:out_num],out_src[:out_num]])
    shape = out_dst.shape
    newNodeSRC = torch.zeros(shape,dtype=torch.int32).to('cuda:0')
    newNodeDST = torch.zeros(shape,dtype=torch.int32).to('cuda:0')
    edgeNUM = seed_num*fanout - count
    unique = torch.zeros(shape,dtype=torch.int32).to('cuda:0')
    uniqueNUM = torch.Tensor([0]).to(torch.int64).to('cuda:0')

    print(all_node.shape)
    signn.torch_graph_mapping(all_node,out_src,out_dst,newNodeSRC,newNodeDST,unique,edgeNUM,uniqueNUM)
    print("newNodeSRC :",newNodeSRC)
    print("newNodeDST : ",newNodeDST)
    print("unique : ",unique)
    print("uniqueNUM : ",uniqueNUM)



if __name__ == "__main__":
    graphEdge = []
    boundList = []
    graphEdge,boundList = haloTest(graphEdge,boundList)
    right_Test(graphEdge,boundList)