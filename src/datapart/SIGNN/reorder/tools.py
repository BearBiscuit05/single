import numpy as np
import os
import torch
import dgl
import gc

def bin2tensor(filePath, datatype=np.int64):
    tensor = np.fromfile(filePath, dtype=datatype)
    return tensor

def emptyCache():
    torch.cuda.empty_cache()
    gc.collect()

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

def convert_to_tensor(data, dtype=torch.int32):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    else:
        return data.to(dtype)

def cooTocsr(srcList,dstList,sliceNUM=1,device=torch.device('cpu')):
    # compact src
    emptyCache()    # empty GPU
    dstList = dstList.cuda()
    inptr = torch.cat([torch.Tensor([0]).to(torch.int32).to(dstList.device),torch.cumsum(torch.bincount(dstList), dim=0)]).to(torch.int32)
    indice = torch.zeros_like(srcList).to(torch.int32).cuda()
    addr = inptr.clone()[:-1].cuda()
    if sliceNUM == 1:
        srcList = srcList.cuda()
        dgl.cooTocsr(inptr,indice,addr,dstList,srcList) # TODO 压缩的函数的第四个参数，所以将dst与src顺序倒换
        inptr = inptr.cpu() 
        indice = indice.cpu()
        addr = None
        srcList = srcList.cpu()
        dstList = dstList.cpu()
        return inptr,indice
    else:
        dstList = dstList.cpu()
        src_batches = torch.chunk(srcList, sliceNUM, dim=0)
        dst_batches = torch.chunk(dstList, sliceNUM, dim=0)
        batch = [src_batches, dst_batches]
        for _,(src_batch,dst_batch) in enumerate(zip(*batch)):
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            dgl.cooTocsr(inptr,indice,addr,dst_batch,src_batch) # compact dst save src
        addr,dst_batch,src_batch= None,None,None
        inptr = inptr.cpu() 
        indice = indice.cpu()
        return inptr,indice

def remapEdgeId(uniTable,srcList,dstList,device=torch.device('cpu')):
    index = torch.arange(len(uniTable)).to(torch.int32)
    remap = torch.zeros(torch.max(uniTable)+1).to(torch.int32)
    # 构建ramap表
    remap[uniTable.to(torch.int64)] = index
    uniTable = uniTable.cpu()
    remap = remap.to(device)
    if srcList != None:
        srcList = srcList.to(device)
        srcList = remap[srcList.to(torch.int64)]
        srcList = srcList.cpu()
    if dstList != None:
        dstList = dstList.to(device)
        dstList = remap[dstList.to(torch.int64)]
        dstList = dstList.cpu()
    return srcList,dstList

def coo2csr_sort(row,col):
    sort_row,indice = torch.sort(row,dim=0)
    indice = col[indice]
    inptr = torch.cat([torch.Tensor([0]).to(torch.int32),torch.cumsum(torch.bincount(sort_row), dim=0)])
    return inptr,indice

def coo2csr_dgl(srcs,dsts):
    g = dgl.graph((dsts,srcs)).formats('csr')       # 顺序倒换，等同于转换CSC，压缩dst
    indptr, indices, _ = g.adj_sparse(fmt='csr')
    return indptr,indices


def countMemToLoss(edgeNUM,nodeNUM,featLen,ratedMem,printInfo=False):
    int32Byte, int64Byte, float32Byte = 4, 8, 4
    MemNeed_B = (edgeNUM + nodeNUM + 1) * int32Byte + nodeNUM * featLen * float32Byte + nodeNUM * int64Byte
    MemNeed_KB = MemNeed_B / (1024 ** 1)
    MemNeed_MB = MemNeed_B / (1024 ** 2)
    MemNeed_GB = MemNeed_B / (1024 ** 3)
    if printInfo:
        print(f"Basic Parameters:")
        print(f"Number of Edges: {edgeNUM},Number of Nodes: {nodeNUM},Feature Length: {featLen}")
        print(f"Memory Needed: {MemNeed_GB:.2f} GB/{MemNeed_MB:.2f} MB/{MemNeed_KB:.2f} KB/{MemNeed_B:.2f} B")
    if ratedMem >= MemNeed_B:
        return False
    else:
        return True
    
def print_gpu_memory(index):
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(index)
        memory_allocated = torch.cuda.memory_allocated(index) / 1024 ** 3  # 转换为GB
        memory_cached = torch.cuda.memory_cached(index) / 1024 ** 3  # 转换为GB
        print(f"GPU {index}: {gpu}")
        print(f"  Allocated Memory: {memory_allocated:.2f} GB")
        print(f"  Cached Memory: {memory_cached:.2f} GB")
    else:
        print("No GPU available.")