import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import sample_hop
import time
import struct

def val_test(graphEdge,boundList,nodeID,valArray):
    edges = graphEdge[boundList[nodeID]:boundList[nodeID+1]]
    print(valArray)
    print(edges)

if __name__ == "__main__":
    graphEdge = []
    boundList = []

    file_path = './../../srcList.bin'
    graphEdge = np.fromfile(file_path, dtype=np.int32)
    graphEdge = torch.tensor(graphEdge).to('cuda:0')
    file_path = "./../../range.bin"
    boundList = np.fromfile(file_path, dtype=np.int32)
    boundList = torch.tensor(boundList).to('cuda:0')


    # file_path = "./../../range.bin"
    # with open(file_path, 'rb') as file:
    #     while True:
    #         data = file.read(4)
    #         if not data:
    #             break
    #         integer = struct.unpack('i', data)[0]
    #         boundList.append(integer)

    batch = 2
#    fanout = n + n * 10 + n * 10 * 10
    fan1 = 2
    fan2 = 2
    fanout1 = batch * fan1 
    fanout2 = fanout1 * fan2
    # graphEdge = torch.Tensor(graphEdge).to(torch.int).to('cuda:0')
    # boundList = torch.Tensor(boundList).to(torch.int).to('cuda:0')

    nodeNUM = len(boundList) - 1
    edgeNUM = len(graphEdge)
    
    trainlist = [i for i in range(1,1025)]
    #trainlist = torch.randint(0,nodeNUM,(1024,))
#    trainlist, _ = torch.sort(trainlist)
    trainlist = torch.Tensor(trainlist).to(torch.int).to('cuda:0')
    
    outputSRC1 = torch.zeros(fanout1).to(torch.int).to(device="cuda:0")
    outputDST1 = torch.zeros(fanout1).to(torch.int).to(device="cuda:0")
    outputSRC2 = torch.zeros(fanout2).to(torch.int).to(device="cuda:0")
    outputDST2 = torch.zeros(fanout2).to(torch.int).to(device="cuda:0")
    #=======================
    start = time.time()
    sample_hop.torch_launch_sample_2hop(
        outputSRC1,outputDST1,outputSRC2,outputDST2, 
        graphEdge, boundList, trainlist,
        fan1,fan2,batch)
    print(time.time()-start)
    print("outputSRC1:",outputSRC1)
    print("outputDST1:",outputDST1)
    print("outputSRC2:",outputSRC2)
    print("outputDST2:",outputDST2)
    
    #=======================
    #val_test(graphEdge,boundList,trainlist[0],output[0:25])

