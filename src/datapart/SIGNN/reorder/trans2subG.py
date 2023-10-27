import numpy as np
import dgl
import torch
import os
from scipy.sparse import csr_matrix
import copy
import gc
import time

"""
input:
    multi file with edges in this partition (bin)
output:
    每个分区包含
    indptr,indices
    subGFeat
    trainMask
"""
MERGETIME = 0

def bin2tensor(filePath, datatype=np.int64):
    tensor = np.fromfile(filePath, dtype=datatype)
    return tensor

def saveBin(tensor,savePath,addSave=False):
    if addSave :
        with open(savePath, 'ab') as f:
            if isinstance(tensor, torch.Tensor):
                tensor.numpy().tofile(f)
            elif isinstance(tensor, np.ndarray):
                tensor.tofile(f)
    else:
        if isinstance(tensor, torch.Tensor):
            tensor.numpy().tofile(savePath)
        elif isinstance(tensor, np.ndarray):
            tensor.tofile(savePath)

def checkFilePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"file '{path}' exist...")

def loadingFeat(featPath,featLen,nodeNUM=0,useMmap=False):
    if useMmap:
        fpr = np.memmap(featPath, dtype='float32', mode='r', shape=(nodeNUM,featLen))
    else:
        fpr = np.fromfile(featPath,dtype=np.float32).reshape(-1,featLen)
    return fpr
    
def nodeShuffle(raw_node,raw_graph,savePath=None,saveRes=False):
    torch.cuda.empty_cache()
    gc.collect()
    srcs = raw_graph[::2]
    dsts = raw_graph[1::2]
    print(len(srcs))
    raw_node = torch.tensor(raw_node).cuda()
    srcs_tensor = torch.tensor(srcs).cuda()
    dsts_tensor = torch.tensor(dsts).cuda()
    uni = torch.ones(len(raw_node)*2).to(torch.int32).cuda()
    print("begin shuffle...")
    #srcShuffled,dstShuffled,uni = dgl.remappingNode(srcs_tensor,dsts_tensor,uni)
    srcShuffled,dstShuffled,uni = dgl.mapByNodeSet(raw_node,uni,srcs_tensor,dsts_tensor)
    srcs_tensor = srcs_tensor.cpu()
    dsts_tensor = dsts_tensor.cpu()
    srcShuffled = srcShuffled.cpu()
    dstShuffled = dstShuffled.cpu()
    uni = uni.cpu()
    if saveRes:
        graph = torch.stack((srcShuffled,dstShuffled),dim=1)
        graph = graph.reshape(-1).numpy()
        graph.tofile(savePath)
    print("shuffle end...")
    return srcShuffled,dstShuffled,uni

def featMerge(featTable,nodes):
    batch_size = 10
    nodes_batches = torch.chunk(nodes, batch_size, dim=0)
    subFeat = np.zeros((len(nodes),128),dtype=np.float32)
    offset = 0
    for nodebatch in nodes_batches:
        subFeat[offset:offset+len(nodebatch)] = featTable[nodebatch.numpy()]
        offset += len(nodebatch)
    return subFeat

def trainIdxSubG(subGNode,trainSet):
    trainSet = torch.tensor(trainSet).to(torch.int32)
    Lid = torch.zeros_like(trainSet).to(torch.int32).cuda()
    dgl.mapLocalId(subGNode.cuda(),trainSet.cuda(),Lid)
    Lid = Lid.cpu()
    return Lid

def coo2csrFromFile(graphbinPath,savePath=None,saveRes=False):
    edges = np.fromfile(graphbinPath,dtype=np.int32)
    row = edges[::2]
    col = edges[1::2]
    data = np.ones(len(col),dtype=np.int32)
    m = csr_matrix((data, (row, col)))
    indptr = m.indptr
    indices = m.indices
    return indptr,indices
    # g = dgl.graph((src, dst))
    # g = g.formats('csc')
    # indptr, indices, _ = g.adj_sparse(fmt='csc')

def coo2csr(srcs,dsts):
    row,col = srcs,dsts
    data = np.ones(len(col),dtype=np.int32)
    m = csr_matrix((data, (row.numpy(), col.numpy())))
    return m.indptr,m.indices

def rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen):
    global MERGETIME
    labels = np.fromfile(LABELPATH,dtype=np.int64)
    for i in range(partitionNUM):
        startTime = time.time()
        PATH = RAWDATAPATH + f"/part{i}" 
        rawDataPath = PATH + f"/raw_G.bin"
        rawTrainPath = PATH + f"/raw_trainIds.bin"
        rawNodePath = PATH + f"/raw_nodes.bin"
        SubFeatPath = PATH + "/feat.bin"
        SubTrainIdPath = PATH + "/trainIds.bin"
        SubIndptrPath = PATH + "/indptr.bin"
        SubIndicesPath = PATH + "/indices.bin"
        SubLabelPath = PATH + "/labels.bin"
        # SubUniPath = PATH + "/GidMap.bin"
        checkFilePath(PATH)
        data = np.fromfile(rawDataPath,dtype=np.int32)
        node = np.fromfile(rawNodePath,dtype=np.int32)
        trainidx = np.fromfile(rawTrainPath,dtype=np.int64)
        srcShuffled,dstShuffled,uni = nodeShuffle(node,data)
        #subfeat = featMerge(fpr,uni)
        subLabel = labels[uni.to(torch.int64)]
        indptr, indices = coo2csr(srcShuffled,dstShuffled)
        trainidx = trainIdxSubG(uni,trainidx)
        #saveBin(uni,SubUniPath)
        saveBin(subLabel,SubLabelPath)
        saveBin(trainidx,SubTrainIdPath)
        #saveBin(subfeat,SubFeatPath)
        saveBin(indptr,SubIndptrPath)
        saveBin(indices,SubIndicesPath)
        print(f"subG_{i} success processed...")
        MERGETIME += time.time() - startTime


# ===============
def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    featPath = "/home/bear/workspace/single-gnn/data/raid/papers100M/feats.bin"
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = np.fromfile(featPath, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset)
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


def genSubGFeat(SAVEPATH,partNUM,nodeNUM,sliceNUM,featLen):
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
        #print("idsSliceList:",idsSliceList[i])
    
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice("",beginIdx,endIdx,featLen)
        for index in range(partNUM):
            fileName = SAVEPATH + f"/part{index}/feats.bin"
            SubIdsList = idsSliceList[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList]
            saveBin(subFeat,fileName,addSave=sliceIndex)

if __name__ == '__main__':
    RAWDATAPATH = "/home/bear/workspace/single-gnn/data/partition/PA"
    FEATPATH = "/home/bear/workspace/single-gnn/data/raid/papers100M/feats.bin"
    SAVEPATH = "/home/bear/workspace/single-gnn/data/partition/PA"
    LABELPATH = "/home/bear/workspace/single-gnn/data/raid/papers100M/labels.bin"
    partitionNUM = 8
    nodeNUM = 111059956
    # featLen = 128
    # rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen)
    # print(f"all do cost time{MERGETIME:.3f}...")
    genSubGFeat(SAVEPATH,partitionNUM,nodeNUM,5,128)