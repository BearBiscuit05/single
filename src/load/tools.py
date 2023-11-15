import torch
import dgl
import numpy as np
import time
from memory_profiler import profile

def featSlice(raw_feat,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    #subFeat = np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset).reshape(-1,featLen)
    subFeat = raw_feat[beginIndex:endIndex]
    return subFeat

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

def countMemToLoss(edgeNUM,nodeNUM,featLen,ratedMem):
    int32Byte = 4
    int64Byte = 8
    float32Byte = 4
    MemNeed = (edgeNUM + nodeNUM + 1) * int32Byte + nodeNUM * featLen * float32Byte + nodeNUM * int64Byte
    MemNeed_GB = MemNeed / (1024 ** 3)
    #print(f"Estimated Memory Needed: {MemNeed_GB:.2f} GB")
    if ratedMem >= MemNeed:
        return False
    else:
        return True
#@profile
def loss_csr(raw_ptr,raw_indice,lossNode,saveNode):
    nodeNUM = raw_ptr.shape[0] - 1
    ptr_diff = torch.diff(raw_ptr).cuda()
    
    if lossNode == None:
        num_elements = int(raw_ptr.shape[0]*0.9)
        min_value = 0
        max_value = nodeNUM
        node_save_idx = torch.randint(min_value, max_value - 1, (num_elements,), dtype=torch.int32)
        
        node_save_idx = torch.unique(node_save_idx)
        node_save_idx,_ = torch.sort(node_save_idx)  
        mask = torch.zeros(ptr_diff.size(0), dtype=torch.bool)
        mask[node_save_idx.to(torch.int64)] = True
    else:
        mask = torch.ones(nodeNUM, dtype=torch.bool).cuda()
        mask[lossNode.to(torch.int64)] = False
        node_save_idx = saveNode

    ptr_diff[lossNode.to(torch.int64)] = 0
    # condition = ptr_diff > 100
    # ptr_diff[condition] = (ptr_diff[condition] * 0.7).to(torch.int32) 
    
    new_ptr = torch.cat((torch.zeros(1).to(torch.int32).cuda(),torch.cumsum(ptr_diff,dim = 0).to(torch.int32))).cuda()
    id2featMap = mask.cumsum(dim=0).to(torch.int32).cuda()
    id2featMap -= 1
    id2featMap[lossNode.to(torch.int64)] = -1
    ptr_diff,mask = None,None

    
    # indice
    new_indice = raw_indice.clone()[:new_ptr[-1].item()]
    dgl.loss_csr(raw_ptr,new_ptr,raw_indice,new_indice)
    raw_ptr,raw_indice = None,None
    return new_ptr,new_indice,id2featMap


#@profile
def loss_feat(loss_feat,raw_feat, sliceNUM, id2featMap, featLen):
    # from cup to gpu with loss
    #print('-'*20)
    #start_preprocess = time.time()
    node_save_idx = torch.nonzero(id2featMap.cpu() >= 0).reshape(-1).to(torch.int32)
    boundList = genSliceBound(sliceNUM, node_save_idx.numel())
    idsSliceList = sliceIds(node_save_idx, boundList)
    #print(f"Preprocess time: {time.time() - start_preprocess : .4f} seconds")
    
    # start_preprocess = time.time()
    #start_loss_feat = time.time()
    # print(f"loss_feat creat time: {time.time() - start_preprocess : .4f} seconds")
    
    offset = 0
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex + 1]
        
        #start_slice = time.time()
        
        sliceFeat = torch.from_numpy(featSlice(raw_feat, beginIdx, endIdx, featLen))
        print(sliceFeat.dtype)
        choice_ids = idsSliceList[sliceIndex] - boundList[sliceIndex]
        sliceSize = choice_ids.shape[0]
        loss_feat[offset:offset + sliceSize] = sliceFeat[choice_ids.to(torch.int64)].cuda()
        offset += sliceSize
        #print(f"Slice {sliceIndex + 1} time: {time.time() - start_slice : .4f} seconds")
    
    #print(f"Total feat slice time: {time.time() - start_loss_feat : .4f} seconds")
    #print('-'*20)
    # return loss_feat
