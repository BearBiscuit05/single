import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import sample_hop
import time
import struct
import os

def val_test(graphEdge,boundList,nodeID,valArray):
    edges = graphEdge[boundList[nodeID]:boundList[nodeID+1]]
    print(valArray)
    print(edges)

def sgnn_train_test(datasetPath,partID,batch,layer,fan,cudaDeviceIndex = 0):

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #调试用
    # layer : 层数
    # batch : 训练节点数
    # fan[0,layer) : 每个节点的1/2/3跳邻居采样数
    # cudaDeviceIndex ：使用哪一张GPU训练
    
    if layer != 1 and layer != 2 and layer != 3:
        return
    
    # 判定类型为list，长度2以上，防止越界
    if isinstance(fan,list) == False or len(fan) < layer:
        print('error fan')
        return

    # 设定GPU设备
    if torch.cuda.is_available() == False:
        print('No GPU Device')
        return
    elif cudaDeviceIndex < 0 or cudaDeviceIndex >= torch.cuda.device_count():
        print('Wrong GPU Index Argument %d, Select default 0'%cudaDeviceIndex)
        cudaDeviceIndex = 0
    deviceName = 'cuda:%d' % cudaDeviceIndex

    graphFile = '%s/part%d/srcList.bin' % (datasetPath,partID)
    graphEdge = torch.tensor(np.fromfile(graphFile, dtype=np.int32)).to(device=deviceName)
    rangeFile = '%s/part%d/range.bin' % (datasetPath,partID)
    boundList = torch.tensor(np.fromfile(rangeFile, dtype=np.int32)).to(device=deviceName)

    fanout = [batch]
    for l in range(0,layer):
        fanout.append(fanout[-1] * fan[l])

    nodeNUM = len(boundList) - 1
    edgeNUM = len(graphEdge)

    trainlist = [i for i in range(1,1025)]
    trainlist = torch.Tensor(trainlist).to(torch.int).to(device=deviceName)

    outputSRC = []
    outputDST = []
    for i in range(0,layer):
        outputSRC.append(torch.zeros(fanout[i+1]).to(torch.int).to(device=deviceName))
        outputDST.append(torch.zeros(fanout[i+1]).to(torch.int).to(device=deviceName))

    start = time.time()

    if layer == 3:
        sample_hop.torch_launch_sample_3hop(
            outputSRC[0],outputDST[0],outputSRC[1],outputDST[1],outputSRC[2],outputDST[2],
            graphEdge, boundList, trainlist,
            fan[0],fan[1],fan[2],batch,cudaDeviceIndex)
    elif layer == 2:
        sample_hop.torch_launch_sample_2hop_new(
            outputSRC[0],outputDST[0],outputSRC[1],outputDST[1],
            graphEdge, boundList, trainlist,
            fan[0],fan[1],batch,cudaDeviceIndex)
    elif layer == 1:
        sample_hop.torch_launch_sample_1hop(
            outputSRC[0],outputDST[0],
            graphEdge, boundList, trainlist,
            fan[0],batch,cudaDeviceIndex)

    end = time.time()

    print("training time : %g second" % (end-start))

    for i in range(0,layer):
        print("outputSRC%d:" % (i+1),outputSRC[i])
        print("outputDST%d:" % (i+1),outputDST[i])
    #=======================
    #val_test(graphEdge,boundList,trainlist[0],output[0:25])

if __name__ == "__main__":
    dataset = '/home/bear/workspace/singleGNN/data/products_4'
    sgnn_train_test(datasetPath=dataset,partID=0,batch=2,layer=1,fan=[2],cudaDeviceIndex=0)
    sgnn_train_test(datasetPath=dataset,partID=1,batch=1024,layer=1,fan=[25],cudaDeviceIndex=1)
    sgnn_train_test(datasetPath=dataset,partID=2,batch=2,layer=2,fan=[2,2],cudaDeviceIndex=2)
    sgnn_train_test(datasetPath=dataset,partID=3,batch=1024,layer=2,fan=[25,10],cudaDeviceIndex=3)
    sgnn_train_test(datasetPath=dataset,partID=2,batch=2,layer=3,fan=[2,2,2],cudaDeviceIndex=4)
    sgnn_train_test(datasetPath=dataset,partID=0,batch=1024,layer=3,fan=[25,10,5],cudaDeviceIndex=-1)
