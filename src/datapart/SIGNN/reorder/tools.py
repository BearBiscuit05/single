import numpy as np
import os
import torch
import dgl
import gc
import time

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

def cooTocsc(srcList,dstList,sliceNUM=1,device=torch.device('cpu')):
    # dstList = dstList.cuda()
    max_value = max(torch.max(dstList).item(), torch.max(srcList).item()) + 1   # 保证对齐
    startTime = time.time()
    binAns = torch.bincount(dstList, minlength=max_value)
    ptrcum = torch.cumsum(binAns.cuda(), dim=0)
    zeroblock=torch.zeros(1,device=ptrcum.device)
    inptr = torch.cat([zeroblock,ptrcum]).to(torch.int32).cuda()
    
    indice = torch.zeros_like(srcList,dtype=torch.int32,device="cuda")
    addr = inptr.clone()[:-1]
    if sliceNUM <= 1:
        dstList = dstList.cuda()
        srcList = srcList.cuda()
        dgl.cooTocsr(inptr,indice,addr,dstList,srcList) # compact dst , exchange place
        inptr,indice = inptr.cpu(),indice.cpu()
        addr = None
        srcList = srcList.cpu()
        dstList = dstList.cpu()
        return inptr,indice
    else:
        src_batches = torch.chunk(srcList, sliceNUM, dim=0)
        dst_batches = torch.chunk(dstList, sliceNUM, dim=0)
        batch = [src_batches, dst_batches]
        for _,(src_batch,dst_batch) in enumerate(zip(*batch)):
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            dgl.cooTocsr(inptr,indice,addr,dst_batch,src_batch) # compact dst , exchange place
        addr,dst_batch,src_batch= None,None,None
        inptr = inptr.cpu() 
        indice = indice.cpu()
        return inptr,indice

def remapEdgeId(uniTable,srcList,dstList,device=torch.device('cpu'),remap=None):
    if remap == None:
        # 构建ramap表
        index = torch.arange(len(uniTable),dtype=torch.int32,device=device)
        remap = torch.zeros(torch.max(uniTable)+1,dtype=torch.int32,device=device)
        remap[uniTable.to(torch.int64)] = index
    uniTable = uniTable.cpu()
    if srcList != None:
        srcList = srcList.to(device)
        srcList = remap[srcList.to(torch.int64)]
        srcList = srcList.cpu()
    if dstList != None:
        dstList = dstList.to(device)
        dstList = remap[dstList.to(torch.int64)]
        dstList = dstList.cpu()
    return srcList,dstList,remap

def coo2csc_sort(row,col):  # src,dst
    sort_col,indice = torch.sort(col,dim=0)
    indice = row[indice]
    inptr = torch.cat([torch.Tensor([0]).to(torch.int32),torch.cumsum(torch.bincount(sort_col), dim=0)])
    return inptr,indice

def coo2csc_dgl(srcs,dsts):
    g = dgl.graph((srcs,dsts)).formats('csc')       # 顺序倒换，等同于转换CSC，压缩dst
    indptr, indices, _ = g.adj_sparse(fmt='csc')
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