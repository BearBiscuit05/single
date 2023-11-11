import numpy as np
import dgl
import torch
import copy
import os
import copy
import gc
import time
from scipy.sparse import csr_matrix,coo_matrix
import json
from tools import *

RUNTIME = 0
SAVETIME = 0
MERGETIME = 0
MAXEDGE = 100000000

def PRgenG(RAWPATH,nodeNUM,partNUM,savePath=None):
    GRAPHPATH = RAWPATH + "/graph.bin"
    TRAINPATH = RAWPATH + "/trainIds.bin"
    FEATPATH = RAWPATH + "/feat.bin"

    checkFilePath(savePath)
    DataPath = savePath + f"/raw_G.bin"
    TrainPath = savePath + f"/raw_trainIds.bin"
    NodePath = savePath + f"/raw_nodes.bin"
    PRvaluePath = savePath + f"/raw_value.bin"

    graph = torch.tensor(np.fromfile(GRAPHPATH,dtype=np.int32))
    src,dst = graph[::2],graph[1::2]
    trainIds = torch.tensor(np.fromfile(TRAINPATH,dtype=np.int64))
    edgeTable = torch.zeros_like(src).to(torch.int32)
    template_array = torch.zeros(nodeNUM,dtype=torch.int32)

    inNodeTable = copy.deepcopy(template_array)
    outNodeTable = copy.deepcopy(template_array)
    inNodeTable,outNodeTable = dgl.sumDegree(inNodeTable.cuda(),outNodeTable.cuda(),src.cuda(),dst.cuda())
    inNodeTable = inNodeTable.cpu()
    outNodeTable = outNodeTable.cpu()

    nodeValue = copy.deepcopy(template_array)
    nodeInfo = copy.deepcopy(template_array)
    nodeValue[trainIds] = 10000

    # random method
    shuffled_indices = torch.randperm(trainIds.size(0))
    r_trainId = trainIds[shuffled_indices]
    trainBatch = torch.chunk(r_trainId, partNUM, dim=0)

    for index,ids in enumerate(trainBatch):
        info = 1 << index
        nodeInfo[ids] = info
        # 存储训练集
        saveBin(ids,TrainPath)
    
    dst = dst.cuda()
    src = src.cuda()
    edgeTable, inNodeTable = edgeTable.cuda(), inNodeTable.cuda()
    nodeValue, nodeInfo = nodeValue.cuda(), nodeInfo.cuda()
    for _ in range(3):    
        dgl.per_pagerank(dst,src,edgeTable,inNodeTable,nodeValue,nodeInfo)
    dst,src = dst.cpu(),src.cpu()
    edgeTable,inNodeTable = edgeTable.cpu(),inNodeTable.cpu()
    nodeValue,nodeInfo = nodeValue.cpu(),nodeInfo.cpu()

    for bit_position in range(partNUM):
        nodeIndex = (nodeInfo & (1 << bit_position)) != 0
        edgeIndex = (edgeTable & (1 << bit_position)) != 0
        nid = torch.nonzero(nodeIndex).reshape(-1)
        eid = torch.nonzero(edgeIndex).reshape(-1)
        graph = graph.reshape(-1,2)
        subEdge = graph[eid]
        partValue = nodeValue[nid]    
        selfLoop = np.repeat(nid.to(torch.int32), 2)
        saveBin(nid,NodePath)
        saveBin(selfLoop,DataPath)
        saveBin(subEdge,DataPath,addSave=True)
        saveBin(partValue,PRvaluePath)

def processSubG(partid):
    partIndex = 0
    partid = idList2part[partIndex]
    partValue = prValue2part[partIndex]
    partFeat = feat2part[partIndex]
    subG = edge2part[partIndex]
    prValue, indices = torch.sort(partValue, descending=True)
    partid = partid[indices]
    partFeat = partFeat[indices]

    srcList = copy.deepcopy(subG[:,0]).cuda()
    dstList = copy.deepcopy(subG[:,1]).cuda()
    uniTable = torch.zeros_like(partid,dtype=torch.int32).cuda()
    partid = partid.to(torch.int32).cuda()
    srcList,dstList,uniTable = dgl.mapByNodeSet(partid,uniTable,srcList,dstList)
    srcList,dstList,uniTable = srcList.cpu(),dstList.cpu(),uniTable.cpu()

    sort_dstList,indice = torch.sort(dstList,dim=0) # 有待提高
    sort_srcList = srcList[indice]

    nodeSize = uniTable.shape[0]
    edgeSize = srcList.shape[0]
    fix_NUM = int(nodeSize * 0.1)
    position = torch.searchsorted(sort_dstList, fix_NUM)

    #===
    fix_indice = sort_srcList[:position]
    fix_inptr = torch.cat([torch.Tensor([0]).to(torch.int32),torch.cumsum(torch.bincount(sort_dstList[:position]), dim=0)]).to(torch.int32)
    random_dst = sort_dstList[position:]
    random_src = sort_srcList[position:]

    mapTable = torch.zeros_like(uniTable).to(torch.int32) - 1
    fix_index = torch.arange(fix_NUM-1).to(torch.int32)
    mapTable[fix_index.to(torch.int64)] = fix_index

    # random choice method
    choice_src = random_src[30000000:60000000]
    choice_dst = random_dst[30000000:60000000]

    choice_ids = torch.nonzero(torch.bincount(choice_dst)).reshape(-1)
    choice_index = torch.arange(len(choice_ids)).to(torch.int32) + fix_NUM
    mapTable[choice_ids] = choice_index
    cumList = torch.bincount(choice_dst)[choice_ids]

    choice_indice = choice_src
    choice_inptr = torch.cumsum(cumList, dim=0).to(torch.int32) + fix_inptr[-1]

    inptr = torch.cat([fix_inptr,choice_inptr])
    indice = torch.cat([fix_indice,choice_indice])

    return inptr,indice,mapTable