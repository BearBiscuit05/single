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

if __name__ == "__main__":
    graphEdge = []
    boundList = []

    file_path = './../../data/products_4/part0/srcList.bin'
    graphEdge1 = np.fromfile(file_path, dtype=np.int32)
    graphEdge1 = torch.tensor(graphEdge1).to('cuda:0')
    file_path = "./../../data/products_4/part0/range.bin"
    boundList1 = np.fromfile(file_path, dtype=np.int32)
    boundList1 = torch.tensor(boundList1).to('cuda:0')
    # print("nodeNUM graph1:",len(boundList1)/2)
    nodeNUM = int((len(boundList1))/2)
    edgeLen = len(graphEdge1)

    file_path = './../../data/products_4/part3/srcList.bin'
    graphEdge2 = np.fromfile(file_path, dtype=np.int32)
    graphEdge2 = torch.tensor(graphEdge2).to('cuda:0')
    file_path = "./../../data/products_4/part3/range.bin"
    boundList2 = np.fromfile(file_path, dtype=np.int32)
    boundList2 = torch.tensor(boundList2).to('cuda:0')
    graphEdge2 = graphEdge2 + int(nodeNUM)
    boundList2 = boundList2 + int(edgeLen)
    # print("nodeNUM graph2:",len(boundList2)/2)

    graphEdge = torch.cat([graphEdge1,graphEdge2])
    boundList = torch.cat([boundList1,boundList2])

    file_path = './../../data/products_4/part0/halo3.bin'
    halo2 = np.fromfile(file_path, dtype=np.int32)
    halo2 = torch.tensor(halo2).to('cuda:0')
    file_path = "./../../data/products_4/part0/halo3_bound.bin"
    halobound = np.fromfile(file_path, dtype=np.int32)
    halobound = torch.tensor(halobound).to('cuda:0')
    # print("halo nodeNUM:",len(halobound)-1)
    graphEdge_tmp = copy.deepcopy(graphEdge).to('cpu')
    start = time.time()
    signn.torch_graph_halo_merge(graphEdge,boundList,halo2,halobound,nodeNUM)
    print("comput time:",time.time()-start)
    graphEdge = graphEdge.to('cpu')
    are_equal = torch.equal(graphEdge, graphEdge_tmp)
    print("same : ",are_equal)
