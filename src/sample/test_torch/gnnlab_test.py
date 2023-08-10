import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import signn
import time
import struct

def val_test(graphEdge,boundList,nodeID,valArray):
    edges = graphEdge[boundList[nodeID]:boundList[nodeID+1]]
    print(valArray)
    print(edges)

if __name__ == "__main__":
    graphEdge = []
    boundList = []

    file_path = './../../data/products_4/part0/srcList.bin'
    graphEdge = np.fromfile(file_path, dtype=np.int32)
    graphEdge = torch.tensor(graphEdge).to('cuda:0')
    file_path = "./../../data/products_4/part0/range.bin"
    boundList = np.fromfile(file_path, dtype=np.int32)
    boundList = torch.tensor(boundList).to('cuda:0')


#     batch = 2
# #    fanout = n + n * 10 + n * 10 * 10
#     fan1 = 2
#     fan2 = 2
#     fanout1 = batch * fan1 
#     fanout2 = fanout1 * fan2
    seed_num = 1024 * 25
    seed = [i for i in range(seed_num)]
    graphEdge = torch.Tensor(graphEdge).to(torch.int).to('cuda:0')
    boundList = torch.Tensor(boundList).to(torch.int).to('cuda:0')
    seed = torch.Tensor(seed).to(torch.int).to('cuda:0')
    
    fanout = 10
    out_src = [-1 for i in range(seed_num*fanout)]
    out_src = torch.Tensor(out_src).to(torch.int).to('cuda:0')
    out_dst = [-1 for i in range(seed_num*fanout)]
    out_dst = torch.Tensor(out_dst).to(torch.int).to('cuda:0')
    # bound,graphEdge,sampleSeed,seed_num,fanout,out_src,out_dst,randomstate
    start = time.time()
    signn.torch_sample_2hop(boundList,graphEdge,seed,seed_num,fanout,out_src,out_dst)
    print("comput time:",time.time()-start)
    # print(out_src)
    # print(out_dst)
