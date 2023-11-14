import torch
import dgl
import numpy as np
import time

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

def genSliceBound(sliceNUM,nodeNUM):
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM
    return boundList

def countMem(edgeNUM,nodeNUM,featLen):
    int32Byte = 4
    int64Byte = 8
    float32Byte = 4
    MemNeed = edgeNUM * 2 * int32Byte + nodeNUM * featLen * float32Byte + nodeNUM * int64Byte
    MemNeed_GB = MemNeed / (1024 ** 3)
    print(f"Estimated Memory Needed: {MemNeed_GB:.2f} GB")
    return MemNeed


