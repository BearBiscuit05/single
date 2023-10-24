import numpy as np
import dgl
import torch
import os
from scipy.sparse import csr_matrix
"""
input:
    multi file with edges in this partition (bin)
output:
    每个分区包含
    indptr,indices
    subGFeat
    trainMask
"""
def bin2tensor(filePath, datatype=np.int64):
    tensor = np.fromfile(filePath, dtype=datatype)
    return tensor

def saveBin(tensor,savePath):
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
    
def nodeShuffle(raw_graph,savePath=None,saveRes=False):
    srcs = raw_graph[::2]
    dsts = raw_graph[1::2]
    srcs_tensor = torch.Tensor(srcs).to(torch.int32).cuda()
    dsts_tensor = torch.Tensor(dsts).to(torch.int32).cuda()
    uni = torch.ones(len(dsts)*2).to(torch.int32).cuda()
    srcShuffled,dstShuffled,uni = dgl.remappingNode(srcs_tensor,dsts_tensor,uni)
    srcShuffled = srcShuffled.cpu()
    dstShuffled = dstShuffled.cpu()
    uni = uni.cpu()
    if saveRes:
        graph = torch.stack((srcShuffled,dstShuffled),dim=1)
        graph = graph.reshape(-1).numpy()
        graph.tofile(savePath)
    return srcShuffled,dstShuffled,uni

def featMerge(featTable,nodes,savePath=None,saveRes=False):
    subFeat = featTable[nodes]
    if saveRes:
        subFeat.tofile(savePath)
    return subFeat

def trainIdxSubG(subGNode,trainSet):
    localTrainId = torch.full(trainSet.shape, -1, dtype=torch.long)
    mask = (subGNode[:, None] == trainSet)
    table = mask.nonzero().reshape(-1)
    col_indices = table[::2]
    row_indices = table[1::2]
    localTrainId[row_indices] = col_indices
    localTrainId = localTrainId[localTrainId >= 0]
    return localTrainId

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
    m = csr_matrix((data, (row, col)))
    return m.indptr,m.indices

def rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,SAVEPATH):
    fpr = loadingFeat(FEATPATH,100)
    for i in range(partitionNUM):
        rawDataPath = RAWDATAPATH + f"/partition_{i}.bin"
        PATH = SAVEPATH + f"/part{i}"
        SubFeatPath = SAVEPATH + "/feat.bin"
        SubIndptrPath = SAVEPATH + "/indptr.bin"
        SubIndicesPath = SAVEPATH + "/indices.bin"
        checkFilePath(PATH)
        data = np.fromfile(rawDataPath,dtype=np.int32)
        srcShuffled,dstShuffled,uni = nodeShuffle(data)
        subfeat = featMerge(fpr,uni)
        indptr, indices = coo2csr(srcShuffled,dstShuffled)
        saveBin(subfeat,SubFeatPath)
        saveBin(indptr,SubIndptrPath)
        saveBin(indices,SubIndicesPath)
        print(f"subG_{i} success processed...")



if __name__ == '__main__':
    RAWDATAPATH = "/home/bear/workspace/single-gnn/src/datapart/data"
    FEATPATH = "/home/bear/workspace/single-gnn/data/raid/papers100M/feats.bin"
    SAVEPATH = "/home/bear/workspace/single-gnn/src/datapart/data"
    partitionNUM = 32
    rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,SAVEPATH)
        