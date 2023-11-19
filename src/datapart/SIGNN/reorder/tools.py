import numpy as np
import os
import torch
import dgl
import gc

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

def convert_to_tensor(data, dtype=torch.int32):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    else:
        return data.to(dtype)

def cooTocsr(srcList,dstList,device=torch.device('cpu')):
    # compact src
    torch.cuda.empty_cache()
    gc.collect()
    srcList = srcList.cuda()
    inptr = torch.cat([torch.Tensor([0]).to(torch.int32).to(srcList.device),torch.cumsum(torch.bincount(srcList), dim=0)]).to(torch.int32)
    indice = torch.zeros_like(srcList).to(torch.int32).cuda()
    addr = inptr.clone()[:-1].cuda()
    srcList = srcList.cuda()
    dstList = dstList.cuda()
    dgl.cooTocsr(inptr,indice,addr,dstList,srcList) # compact dst save src
    inptr = inptr.cpu() 
    indice = indice.cpu()
    addr = None
    srcList = srcList.cpu()
    dstList = dstList.cpu()
    return inptr,indice

def remapEdgeId(uniTable,srcList,dstList,device=torch.device('cpu')):
    # uniTable必须是unqiue的
    index = torch.arange(len(uniTable)).to(torch.int32)
    remap = torch.zeros(torch.max(uniTable)+1).to(torch.int32)
    # 构建ramap表
    remap[uniTable.to(torch.int64)] = index
    remap = remap.to(device)
    srcList = srcList.to(device)
    srcList = remap[srcList.to(torch.int64)]
    srcList = srcList.cpu()
    dstList = dstList.to(device)
    dstList = remap[dstList.to(torch.int64)]
    dstList = dstList.cpu()
    return srcList,dstList

def coo2csr_sort(row,col):
    """
        row = torch.Tensor([0,0,1,1,2,3,3]).to(torch.int32)
        col = torch.Tensor([0,1,1,2,0,2,3]).to(torch.int32)
    """
    sort_row,indice = torch.sort(row,dim=0)
    indice = col[indice]
    inptr = torch.cat([torch.Tensor([0]).to(torch.int32),torch.cumsum(torch.bincount(sort_row), dim=0)])
    return inptr,indice