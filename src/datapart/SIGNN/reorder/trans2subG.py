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
    fpr = loadingFeat(FEATPATH,featLen)
    labels = np.fromfile(LABELPATH,dtype=np.int64)
    for i in range(partitionNUM):
        rawDataPath = RAWDATAPATH + f"/part{i}.bin"
        rawTrainPath = RAWDATAPATH + f"/trainIds{i}.bin"
        PATH = SAVEPATH + f"/part{i}"
        SubFeatPath = PATH + "/feat.bin"
        SubTrainIdPath = PATH + "/trainIds.bin"
        SubIndptrPath = PATH + "/indptr.bin"
        SubIndicesPath = PATH + "/indices.bin"
        SubLabelPath = PATH + "/labels.bin"
        checkFilePath(PATH)
        data = np.fromfile(rawDataPath,dtype=np.int32)
        trainidx = np.fromfile(rawTrainPath,dtype=np.int64)
        srcShuffled,dstShuffled,uni = nodeShuffle(data)
        subfeat = featMerge(fpr,uni)
        subLabel = labels[uni.to(torch.int64)]
        indptr, indices = coo2csr(srcShuffled,dstShuffled)
        trainidx = trainIdxSubG(uni,trainidx)
        saveBin(subLabel,SubLabelPath)
        saveBin(trainidx,SubTrainIdPath)
        saveBin(subfeat,SubFeatPath)
        saveBin(indptr,SubIndptrPath)
        saveBin(indices,SubIndicesPath)
        print(f"subG_{i} success processed...")


if __name__ == '__main__':
    RAWDATAPATH = "/home/bear/workspace/single-gnn/data/partition/PD"
    FEATPATH = "/home/bear/workspace/single-gnn/data/raid/ogbn_products/feat.bin"
    SAVEPATH = "/home/bear/workspace/single-gnn/data/partition/PD/processed"
    LABELPATH = "/home/bear/workspace/single-gnn/data/raid/ogbn_products/labels_64.bin"
    partitionNUM = 4
    featLen = 100
    rawData2GNNData(RAWDATAPATH,partitionNUM,FEATPATH,LABELPATH,SAVEPATH,featLen)