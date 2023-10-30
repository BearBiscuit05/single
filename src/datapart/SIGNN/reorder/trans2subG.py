import numpy as np
import dgl
import torch
import os
import copy
import gc
import time
from scipy.sparse import csr_matrix,coo_matrix
import json
from tools import *


# =============== 1.partition

def acc_ana(tensor):
    num_ones = torch.sum(tensor == 1).item()  
    total_elements = tensor.numel()  
    percentage_ones = (num_ones / total_elements) * 100 
    print(f"only use by one train node : {percentage_ones:.2f}%")
    num_greater_than_1 = torch.sum(tensor > 1).item() 
    percentage_greater_than_1 = (num_greater_than_1 / total_elements) * 100
    print(f"use by multi train nodes : {percentage_greater_than_1:.2f}%")
    # edgeNUM = edgeTable.cpu().sum() - edgeNUM
    # print(f"edge add to subG : {edgeNUM} , {edgeNUM * 1.0 / allEdgeNUM * 100 :.2f}% of total edges")
    # print(f"after {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
    # f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")

RUNTIME = 0
SAVETIME = 0
MERGETIME = 0
MAXEDGE = 100000000
## bfs 遍历获取基础子图
def analysisG(graph,maxID,partID,trainId=None,savePath=None):
    global RUNTIME
    global SAVETIME
    dst = torch.tensor(graph[::2])
    src = torch.tensor(graph[1::2])
    if trainId == None:
        trainId = torch.arange(int(maxID*0.01),dtype=torch.int64)
    nodeTable = torch.zeros(maxID,dtype=torch.int32)
    nodeTable[trainId] = 1

    batch_size = len(src) // MAXEDGE + 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    repeats = 3
    acc = True
    start = time.time()
    edgeTable = torch.zeros_like(src,dtype=torch.int32).cuda()
    edgeNUM = 0
    allEdgeNUM = src.numel()
    for index in range(1,repeats+1):
        acc_tabel = torch.zeros_like(nodeTable,dtype=torch.int32)
        # print(f"before {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
        #     f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")
        offset = 0
        for src_batch,dst_batch in zip(*batch):
            tmp_nodeTabel = copy.deepcopy(nodeTable)
            tmp_nodeTabel = tmp_nodeTabel.cuda()
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            dgl.fastFindNeigEdge(tmp_nodeTabel,edgeTable,src_batch, dst_batch, offset)
            offset += len(src_batch)
            tmp_nodeTabel = tmp_nodeTabel.cpu()
            acc_tabel = acc_tabel | tmp_nodeTabel
        #acc_ana(acc_tabel)
        nodeTable = acc_tabel
    edgeTable = edgeTable.cpu()
    graph = graph.reshape(-1,2)
    nodeSet =  torch.nonzero(nodeTable).reshape(-1).to(torch.int32)
    edgeTable = torch.nonzero(edgeTable).reshape(-1).to(torch.int32)
    selfLoop = np.repeat(nodeSet.to(torch.int32), 2)
    subGEdge = graph[edgeTable]
    RUNTIME += time.time()-start

    saveTime = time.time()
    checkFilePath(savePath)
    DataPath = savePath + f"/raw_G.bin"
    TrainPath = savePath + f"/raw_trainIds.bin"
    NodePath = savePath + f"/raw_nodes.bin"
    saveBin(nodeSet,NodePath)
    saveBin(selfLoop,DataPath)
    saveBin(subGEdge,DataPath,addSave=True)
    saveBin(trainId,TrainPath)
    SAVETIME += time.time()-saveTime
    return RUNTIME,SAVETIME

# =============== 2.graphToSub    
def nodeShuffle(raw_node,raw_graph,savePath=None,saveRes=False):
    torch.cuda.empty_cache()
    gc.collect()
    srcs = raw_graph[1::2]
    dsts = raw_graph[::2]
    raw_node = torch.tensor(raw_node).cuda()
    print(len(srcs))
    print(len(raw_node))
    srcs_tensor = torch.from_numpy(srcs).to(torch.int32)
    dsts_tensor = torch.from_numpy(dsts).to(torch.int32)
    uni = torch.ones(len(raw_node)*2).to(torch.int32).cuda()
    print("begin shuffle...")
    
    batch_size = len(srcs) // MAXEDGE + 1
    src_batches = list(torch.chunk(srcs_tensor, batch_size, dim=0))
    dst_batches = list(torch.chunk(dsts_tensor, batch_size, dim=0))

    batch = [src_batches, dst_batches]
    # print(src_batches)
    # print(dst_batches)
    # print('-'*20)
    for index,(src_batch,dst_batch) in enumerate(zip(*batch)):
        src_batch = src_batch.cuda()
        dst_batch = dst_batch.cuda()
        srcShuffled,dstShuffled,uni = dgl.mapByNodeSet(raw_node,uni,src_batch,dst_batch)
        src_batches[index] = srcShuffled.cpu()
        dst_batches[index] = dstShuffled.cpu()
    srcs_tensor = torch.cat(src_batches)
    dsts_tensor = torch.cat(dst_batches)
    # print(srcs_tensor)
    # print(dsts_tensor)
    # exit(-1)
    uni = uni.cpu()
    if saveRes:
        graph = torch.stack((srcShuffled,dstShuffled),dim=1)
        graph = graph.reshape(-1).numpy()
        graph.tofile(savePath)
    print("shuffle end...")
    return srcs_tensor,dsts_tensor,uni

def trainIdxSubG(subGNode,trainSet):
    trainSet = torch.tensor(trainSet).to(torch.int32)
    Lid = torch.zeros_like(trainSet).to(torch.int32).cuda()
    dgl.mapLocalId(subGNode.cuda(),trainSet.cuda(),Lid)
    Lid = Lid.cpu().to(torch.int64)
    return Lid

def coo2csr(srcs,dsts):
    g = dgl.graph((srcs, dsts)).formats('csr')
    indptr, indices, _ = g.adj_sparse(fmt='csr')
    return indptr,indices
    # row,col = srcs,dsts
    # data = np.ones(len(col),dtype=np.int32)
    # m = csr_matrix((data, (row.numpy(), col.numpy())))
    # return m.indptr,m.indices

def rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen):
    labels = np.fromfile(LABELPATH,dtype=np.int64)
    for i in range(partitionNUM):
        startTime = time.time()
        PATH = RAWDATAPATH + f"/part{i}" 
        rawDataPath = PATH + f"/raw_G.bin"
        rawTrainPath = PATH + f"/raw_trainIds.bin"
        rawNodePath = PATH + f"/raw_nodes.bin"
        SubTrainIdPath = PATH + "/trainIds.bin"
        SubIndptrPath = PATH + "/indptr.bin"
        SubIndicesPath = PATH + "/indices.bin"
        SubLabelPath = PATH + "/labels.bin"
        checkFilePath(PATH)
        data = np.fromfile(rawDataPath,dtype=np.int32)
        node = np.fromfile(rawNodePath,dtype=np.int32)
        trainidx = np.fromfile(rawTrainPath,dtype=np.int64)
        srcShuffled,dstShuffled,uni = nodeShuffle(node,data)
        subLabel = labels[uni.to(torch.int64)]
        indptr, indices = coo2csr(srcShuffled,dstShuffled)
        trainidx = trainIdxSubG(uni,trainidx)
        saveBin(subLabel,SubLabelPath)
        saveBin(trainidx,SubTrainIdPath)
        saveBin(indptr,SubIndptrPath)
        saveBin(indices,SubIndicesPath)

# =============== 3.featTrans
def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset)
    return subFeat.reshape(-1,featLen)

def sliceIds(Ids,sliceTable):
    beginIndex = 0
    ans = []
    for tar in sliceTable[1:]:
        position = torch.searchsorted(Ids, tar)
        slice = Ids[beginIndex:position]
        ans.append(slice)
        beginIndex = position
    return ans

def genSubGFeat(SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # 获得切片
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM
    print("bound:",boundList)

    idsSliceList = [[] for i in range(partNUM)]
    for i in range(partNUM):
        file = SAVEPATH + f"/part{i}/raw_nodes.bin"
        ids = torch.tensor(np.fromfile(file,dtype=np.int32))
        idsSliceList[i] = sliceIds(ids,boundList)
    
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen)
        for index in range(partNUM):
            fileName = SAVEPATH + f"/part{index}/feat.bin"
            SubIdsList = idsSliceList[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList]
            saveBin(subFeat,fileName,addSave=sliceIndex)

if __name__ == '__main__':
    JSONPATH = "/home/bear/workspace/single-gnn/datasetInfo.json"
    partitionNUM = 8
    sliceNUM = 5
    with open(JSONPATH, 'r') as file:
        data = json.load(file)
    datasetName = ["PA"] 
    for NAME in datasetName:
        GRAPHPATH = data[NAME]["rawFilePath"]
        maxID = data[NAME]["nodes"]
        subGSavePath = data[NAME]["processedPath"]
        trainId = torch.tensor(np.fromfile(GRAPHPATH + "/trainIds.bin",dtype=np.int64))
        # trainId = torch.randint(maxID, (int(maxID*0.01),))
        # trainBatch = torch.chunk(trainId, partitionNUM, dim=0)

        shuffled_indices = torch.randperm(trainId.size(0))
        trainId = trainId[shuffled_indices]
        trainBatch = torch.chunk(trainId, partitionNUM, dim=0)

        graph = np.fromfile(GRAPHPATH+"/graph.bin",dtype=np.int32)
        startTime = time.time()
        for index,trainids in enumerate(trainBatch):
            analysisG(graph,maxID,index,trainId=trainids,savePath=subGSavePath+f"/part{index}")
        print(f"run time cost:{RUNTIME:.3f}")
        print(f"save time cost:{SAVETIME:.3f}")
        print(f"partition all cost:{time.time()-startTime:.3f}")
    
    for NAME in datasetName:
        RAWDATAPATH = data[NAME]["processedPath"]
        FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
        LABELPATH = data[NAME]["rawFilePath"] + "/labels.bin"
        SAVEPATH = data[NAME]["processedPath"]
        nodeNUM = data[NAME]["nodes"]
        featLen = data[NAME]["featLen"]
        
        MERGETIME = time.time()
        #rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen)
        print(f"trans graph cost time{time.time() - MERGETIME:.3f}...")
        FEATTIME = time.time()
        genSubGFeat(SAVEPATH,FEATPATH,partitionNUM,nodeNUM,sliceNUM,featLen)
        print(f"graph feat gen cost time{time.time() - FEATTIME:.3f}...")