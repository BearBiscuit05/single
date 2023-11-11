import numpy as np
import os
import torch

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

def coo2csr(row,col):
    """
        row = torch.Tensor([0,0,1,1,2,3,3]).to(torch.int32)
        col = torch.Tensor([0,1,1,2,0,2,3]).to(torch.int32)
    """
    sort_row,indice = torch.sort(row,dim=0)
    indice = col[indice]
    inptr = torch.cat([torch.Tensor([0]).to(torch.int32),torch.cumsum(torch.bincount(sort_row), dim=0)])
    return inptr,indice